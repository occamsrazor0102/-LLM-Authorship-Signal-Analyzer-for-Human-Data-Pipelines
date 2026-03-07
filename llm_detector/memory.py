"""BEET Historical Memory Store.

Unified persistence layer for cross-batch detection memory.
All data lives in a single directory (default .beet/).

Usage:
    store = MemoryStore('.beet/')
    store.record_batch(results, text_map, batch_id='batch_001')
    history = store.get_attempter_history('worker_42')
    cross_matches = store.cross_batch_similarity(results, text_map)
    store.record_confirmation('task_001', 'ai', verified_by='reviewer_A')
"""

import os
import json
import hashlib
import statistics
from datetime import datetime
from collections import defaultdict, Counter
from pathlib import Path

from llm_detector.baselines import _BASELINE_FIELDS
from llm_detector.similarity import _word_shingles, _STRUCT_FEATURES


# ── MinHash utilities (local copies to avoid circular imports) ────────────

def _shingle_fingerprint(shingle_set, n_hashes=128):
    """MinHash fingerprint from shingle set."""
    if not shingle_set:
        return [0] * n_hashes
    minhashes = [float('inf')] * n_hashes
    for shingle in shingle_set:
        shingle_bytes = ' '.join(shingle).encode('utf-8')
        for i in range(n_hashes):
            h = int(hashlib.md5(
                f"{i}:{shingle_bytes.hex()}".encode()
            ).hexdigest()[:8], 16)
            minhashes[i] = min(minhashes[i], h)
    return minhashes


def _minhash_similarity(fp_a, fp_b):
    """Estimate Jaccard similarity from MinHash fingerprints."""
    if not fp_a or not fp_b or len(fp_a) != len(fp_b):
        return 0.0
    return sum(1 for a, b in zip(fp_a, fp_b) if a == b) / len(fp_a)


class MemoryStore:
    """Persistent memory for the BEET detection pipeline."""

    def __init__(self, store_dir='.beet'):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        (self.store_dir / 'calibration_history').mkdir(exist_ok=True)

        self.submissions_path = self.store_dir / 'submissions.jsonl'
        self.fingerprints_path = self.store_dir / 'fingerprints.jsonl'
        self.attempters_path = self.store_dir / 'attempters.jsonl'
        self.confirmed_path = self.store_dir / 'confirmed.jsonl'
        self.calibration_path = self.store_dir / 'calibration.json'
        self.config_path = self.store_dir / 'config.json'

        self._config = self._load_config()
        if not self.config_path.exists():
            self._save_config()

    # ── Config ────────────────────────────────────────────────────

    def _load_config(self):
        if self.config_path.exists():
            with open(self.config_path) as f:
                return json.load(f)
        return {
            'version': '0.66',
            'created': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'total_submissions': 0,
            'total_batches': 0,
            'total_attempters': 0,
            'total_confirmed': 0,
            'occupations': [],
        }

    def _save_config(self):
        self._config['last_updated'] = datetime.now().isoformat()
        with open(self.config_path, 'w') as f:
            json.dump(self._config, f, indent=2)

    # ── Batch Recording ──────────────────────────────────────────

    def record_batch(self, results, text_map, batch_id=None):
        """Record a full batch of pipeline results to memory.

        Updates submissions, fingerprints, attempter profiles, and config.
        """
        if batch_id is None:
            batch_id = f"batch_{datetime.now().strftime('%Y-%m-%d_%H:%M')}"

        timestamp = datetime.now().isoformat()
        n_written = 0

        # Write submissions
        with open(self.submissions_path, 'a') as f:
            for r in results:
                record = self._extract_submission_record(r, batch_id, timestamp)
                f.write(json.dumps(record) + '\n')
                n_written += 1

        # Write fingerprints
        self._write_fingerprints(results, text_map, batch_id)

        # Update attempter profiles
        self._update_attempter_profiles(results, batch_id, timestamp)

        # Update config
        self._config['total_submissions'] += n_written
        self._config['total_batches'] += 1
        occs = set(self._config.get('occupations', []))
        for r in results:
            occ = r.get('occupation', '')
            if occ:
                occs.add(occ)
        self._config['occupations'] = sorted(occs)
        self._save_config()

        print(f"  Memory: {n_written} submissions recorded to {self.store_dir}/")
        return n_written

    def _extract_submission_record(self, r, batch_id, timestamp):
        """Extract storable fields from a pipeline result."""
        record = {k: r.get(k) for k in _BASELINE_FIELDS}
        record['batch_id'] = batch_id
        record['timestamp'] = timestamp
        record['pipeline_version'] = r.get('pipeline_version', 'unknown')

        # Similarity context
        record['similarity_partners'] = r.get('similarity_partners', 0)
        record['similarity_max_jaccard'] = r.get('similarity_max_jaccard', 0.0)
        record['similarity_max_semantic'] = r.get('similarity_max_semantic', 0.0)
        if 'similarity_upgrade' in r:
            record['similarity_upgrade'] = r['similarity_upgrade']

        # Length bin
        wc = r.get('word_count', 0)
        if wc < 100:
            record['length_bin'] = 'short'
        elif wc < 300:
            record['length_bin'] = 'medium'
        elif wc < 800:
            record['length_bin'] = 'long'
        else:
            record['length_bin'] = 'very_long'

        return record

    def _write_fingerprints(self, results, text_map, batch_id):
        """Write MinHash and optional embedding fingerprints."""
        try:
            from llm_detector.compat import HAS_SEMANTIC, get_semantic_models
        except ImportError:
            HAS_SEMANTIC = False

        # Pre-compute embeddings if available
        embeddings = {}
        if HAS_SEMANTIC:
            try:
                embedder, _, _ = get_semantic_models()
                texts = []
                tids = []
                for r in results:
                    tid = r.get('task_id', '')
                    text = text_map.get(tid, '')
                    if text:
                        texts.append(text)
                        tids.append(tid)
                if texts:
                    raw_embeds = embedder.encode(texts)
                    for tid, emb in zip(tids, raw_embeds):
                        embeddings[tid] = [round(float(v), 5) for v in emb[:64]]
            except Exception:
                pass

        with open(self.fingerprints_path, 'a') as f:
            for r in results:
                tid = r.get('task_id', '')
                text = text_map.get(tid, '')
                if not text:
                    continue

                shingles = _word_shingles(text)
                minhash = _shingle_fingerprint(shingles)
                struct_vec = {feat: r.get(feat, 0) for feat in _STRUCT_FEATURES}

                record = {
                    'task_id': tid,
                    'attempter': r.get('attempter', ''),
                    'occupation': r.get('occupation', ''),
                    'batch_id': batch_id,
                    'determination': r.get('determination', ''),
                    'minhash_128': minhash,
                    'structural_vec': struct_vec,
                }
                if tid in embeddings:
                    record['embedding_64'] = embeddings[tid]

                f.write(json.dumps(record) + '\n')

    # ── Attempter Profiles ───────────────────────────────────────

    def _update_attempter_profiles(self, results, batch_id, timestamp):
        """Update rolling attempter profiles with new batch results."""
        profiles = self._load_attempter_profiles()

        by_att = defaultdict(list)
        for r in results:
            att = r.get('attempter', '').strip()
            if att:
                by_att[att].append(r)

        for att, submissions in by_att.items():
            if att not in profiles:
                profiles[att] = {
                    'attempter': att,
                    'total_submissions': 0,
                    'determinations': {'RED': 0, 'AMBER': 0, 'YELLOW': 0,
                                       'GREEN': 0, 'MIXED': 0, 'REVIEW': 0},
                    'confirmed_ai': 0,
                    'confirmed_human': 0,
                    'occupations': [],
                    'batches': [],
                    'first_seen': timestamp,
                    'feature_sums': {},
                    'feature_counts': 0,
                }

            p = profiles[att]
            p['total_submissions'] += len(submissions)
            p['last_seen'] = timestamp
            p['last_updated'] = timestamp

            if batch_id not in p['batches']:
                p['batches'].append(batch_id)

            for r in submissions:
                det = r.get('determination', 'GREEN')
                p['determinations'][det] = p['determinations'].get(det, 0) + 1

                occ = r.get('occupation', '')
                if occ and occ not in p['occupations']:
                    p['occupations'].append(occ)

                for feat in ['prompt_signature_cfd', 'instruction_density_idi',
                             'voice_dissonance_vsd', 'voice_dissonance_spec_score',
                             'self_similarity_nssi_score']:
                    val = r.get(feat, 0)
                    if feat not in p['feature_sums']:
                        p['feature_sums'][feat] = 0.0
                    p['feature_sums'][feat] += val
                p['feature_counts'] += 1

            # Derived fields
            total = p['total_submissions']
            flagged = (p['determinations'].get('RED', 0) +
                       p['determinations'].get('AMBER', 0) +
                       p['determinations'].get('MIXED', 0))
            p['flag_rate'] = round(flagged / max(total, 1), 3)

            if p['feature_counts'] > 0:
                p['mean_features'] = {
                    k: round(v / p['feature_counts'], 3)
                    for k, v in p['feature_sums'].items()
                }

            # Risk tier
            p['risk_tier'] = self._compute_risk_tier(p)

            # Primary detection channel
            channel_counts = Counter()
            for r in submissions:
                if r.get('determination') in ('RED', 'AMBER', 'MIXED'):
                    cd = r.get('channel_details', {}).get('channels', {})
                    for ch, info in cd.items():
                        if info.get('severity') in ('RED', 'AMBER'):
                            channel_counts[ch] += 1
            if channel_counts:
                p['primary_detection_channel'] = channel_counts.most_common(1)[0][0]

        self._save_attempter_profiles(profiles)
        self._config['total_attempters'] = len(profiles)

    @staticmethod
    def _compute_risk_tier(profile):
        """Compute risk tier from flag rate and confirmation history."""
        flag_rate = profile.get('flag_rate', 0)
        confirmed_ai = profile.get('confirmed_ai', 0)

        if confirmed_ai > 0 and flag_rate > 0.50:
            return 'CRITICAL'
        elif flag_rate > 0.30 or confirmed_ai > 0:
            return 'HIGH'
        elif flag_rate > 0.15:
            return 'ELEVATED'
        else:
            return 'NORMAL'

    def _load_attempter_profiles(self):
        """Load attempter profiles dict."""
        profiles = {}
        if self.attempters_path.exists():
            with open(self.attempters_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            p = json.loads(line)
                            profiles[p['attempter']] = p
                        except (json.JSONDecodeError, KeyError):
                            continue
        return profiles

    def _save_attempter_profiles(self, profiles):
        """Save attempter profiles (full rewrite)."""
        with open(self.attempters_path, 'w') as f:
            for p in sorted(profiles.values(),
                            key=lambda x: x.get('flag_rate', 0), reverse=True):
                f.write(json.dumps(p) + '\n')

    # ── Queries ──────────────────────────────────────────────────

    def get_attempter_history(self, attempter):
        """Get full history for a specific attempter."""
        profiles = self._load_attempter_profiles()
        profile = profiles.get(attempter.strip())

        submissions = []
        if self.submissions_path.exists():
            with open(self.submissions_path) as f:
                for line in f:
                    try:
                        rec = json.loads(line.strip())
                        if rec.get('attempter', '').strip() == attempter.strip():
                            submissions.append(rec)
                    except json.JSONDecodeError:
                        continue

        confirmations = []
        if self.confirmed_path.exists():
            with open(self.confirmed_path) as f:
                for line in f:
                    try:
                        rec = json.loads(line.strip())
                        if rec.get('attempter', '').strip() == attempter.strip():
                            confirmations.append(rec)
                    except json.JSONDecodeError:
                        continue

        return {
            'profile': profile,
            'submissions': submissions,
            'confirmations': confirmations,
        }

    def get_attempter_risk_report(self, min_submissions=2):
        """Get all attempters ranked by risk tier and flag rate."""
        profiles = self._load_attempter_profiles()
        tier_order = {'CRITICAL': 4, 'HIGH': 3, 'ELEVATED': 2, 'NORMAL': 1}
        return sorted(
            [p for p in profiles.values()
             if p.get('total_submissions', 0) >= min_submissions],
            key=lambda p: (-tier_order.get(p.get('risk_tier', 'NORMAL'), 0),
                           -p.get('flag_rate', 0)),
        )

    def get_occupation_baselines(self, occupation):
        """Get historical feature distributions for an occupation."""
        submissions = []
        if self.submissions_path.exists():
            with open(self.submissions_path) as f:
                for line in f:
                    try:
                        rec = json.loads(line.strip())
                        if rec.get('occupation', '') == occupation:
                            submissions.append(rec)
                    except json.JSONDecodeError:
                        continue
        return submissions

    # ── Cross-Batch Similarity ───────────────────────────────────

    def cross_batch_similarity(self, current_results, text_map,
                               minhash_threshold=0.50):
        """Compare current batch against historical fingerprints."""
        historical = []
        if self.fingerprints_path.exists():
            with open(self.fingerprints_path) as f:
                for line in f:
                    try:
                        historical.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue

        if not historical:
            return []

        flags = []
        for r in current_results:
            tid = r.get('task_id', '')
            text = text_map.get(tid, '')
            if not text:
                continue

            current_minhash = _shingle_fingerprint(_word_shingles(text))

            for hist in historical:
                if hist.get('task_id') == tid:
                    continue

                att_curr = r.get('attempter', '').strip().lower()
                att_hist = hist.get('attempter', '').strip().lower()
                if att_curr and att_hist and att_curr == att_hist:
                    continue

                mh_sim = _minhash_similarity(
                    current_minhash, hist.get('minhash_128', []))

                if mh_sim >= minhash_threshold:
                    flags.append({
                        'current_id': tid,
                        'historical_id': hist['task_id'],
                        'current_attempter': r.get('attempter', ''),
                        'historical_attempter': hist.get('attempter', ''),
                        'occupation': r.get('occupation', ''),
                        'minhash_similarity': round(mh_sim, 3),
                        'historical_determination': hist.get('determination', '?'),
                        'historical_batch': hist.get('batch_id', '?'),
                    })

        flags.sort(key=lambda f: f['minhash_similarity'], reverse=True)
        return flags

    # ── Confirmation Feedback ────────────────────────────────────

    def record_confirmation(self, task_id, ground_truth, verified_by='',
                            notes=''):
        """Record a human-verified ground truth label."""
        # Find original submission
        original = None
        if self.submissions_path.exists():
            with open(self.submissions_path) as f:
                for line in f:
                    try:
                        rec = json.loads(line.strip())
                        if rec.get('task_id') == task_id:
                            original = rec
                            break
                    except json.JSONDecodeError:
                        continue

        record = {
            'task_id': task_id,
            'ground_truth': ground_truth,
            'verified_by': verified_by,
            'verified_at': datetime.now().isoformat(),
            'notes': notes,
        }
        if original:
            record['attempter'] = original.get('attempter', '')
            record['occupation'] = original.get('occupation', '')
            record['pipeline_determination'] = original.get('determination', '')
            record['pipeline_confidence'] = original.get('confidence', 0)

        with open(self.confirmed_path, 'a') as f:
            f.write(json.dumps(record) + '\n')

        # Update attempter profile
        if original and original.get('attempter'):
            profiles = self._load_attempter_profiles()
            att = original['attempter'].strip()
            if att in profiles:
                if ground_truth == 'ai':
                    profiles[att]['confirmed_ai'] = profiles[att].get(
                        'confirmed_ai', 0) + 1
                else:
                    profiles[att]['confirmed_human'] = profiles[att].get(
                        'confirmed_human', 0) + 1
                profiles[att]['risk_tier'] = self._compute_risk_tier(profiles[att])
                self._save_attempter_profiles(profiles)

        self._config['total_confirmed'] = self._config.get(
            'total_confirmed', 0) + 1
        self._save_config()

        print(f"  Confirmed: {task_id} = {ground_truth} (by {verified_by})")

    # ── Calibration Integration ──────────────────────────────────

    def rebuild_calibration(self):
        """Rebuild calibration table from all confirmed human submissions."""
        from llm_detector.calibration import calibrate_from_baselines, save_calibration
        import shutil
        import tempfile

        # Collect confirmed labels
        confirmed = {}
        if self.confirmed_path.exists():
            with open(self.confirmed_path) as f:
                for line in f:
                    try:
                        rec = json.loads(line.strip())
                        confirmed[rec['task_id']] = rec['ground_truth']
                    except (json.JSONDecodeError, KeyError):
                        continue

        if not confirmed:
            print("  No confirmed labels — cannot rebuild calibration")
            return None

        # Build labeled JSONL for calibration
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl',
                                         delete=False) as tmp:
            if self.submissions_path.exists():
                with open(self.submissions_path) as f:
                    for line in f:
                        try:
                            rec = json.loads(line.strip())
                            tid = rec.get('task_id', '')
                            if tid in confirmed:
                                rec['ground_truth'] = confirmed[tid]
                                tmp.write(json.dumps(rec) + '\n')
                        except json.JSONDecodeError:
                            continue
            tmp_path = tmp.name

        cal = calibrate_from_baselines(tmp_path)
        os.unlink(tmp_path)

        if cal is None:
            print("  Insufficient confirmed human data for calibration")
            return None

        # Snapshot current calibration before overwriting
        if self.calibration_path.exists():
            snapshot_name = f"cal_{datetime.now().strftime('%Y-%m-%d_%H%M')}.json"
            snapshot_path = self.store_dir / 'calibration_history' / snapshot_name
            shutil.copy2(self.calibration_path, snapshot_path)

        save_calibration(cal, str(self.calibration_path))
        return cal

    # ── Pipeline Integration Hook ────────────────────────────────

    def pre_batch_context(self, attempter=None, occupation=None):
        """Retrieve historical context before running a batch."""
        context = {}

        if attempter:
            profiles = self._load_attempter_profiles()
            profile = profiles.get(attempter.strip())
            if profile:
                context['attempter_risk_tier'] = profile.get('risk_tier', 'UNKNOWN')
                context['attempter_flag_rate'] = profile.get('flag_rate', 0)
                context['attempter_total'] = profile.get('total_submissions', 0)
                context['attempter_confirmed_ai'] = profile.get('confirmed_ai', 0)

        if occupation:
            subs = self.get_occupation_baselines(occupation)
            if len(subs) >= 5:
                cfd_values = [s.get('prompt_signature_cfd', 0) for s in subs]
                idi_values = [s.get('instruction_density_idi', 0) for s in subs]
                context['occupation_n'] = len(subs)
                context['occupation_median_cfd'] = round(
                    statistics.median(cfd_values), 3)
                context['occupation_median_idi'] = round(
                    statistics.median(idi_values), 3)

        return context

    # ── Summary ──────────────────────────────────────────────────

    def print_summary(self):
        """Print memory store summary."""
        c = self._config
        print(f"\n  BEET Memory Store: {self.store_dir}/")
        print(f"    Submissions: {c.get('total_submissions', 0)}")
        print(f"    Batches:     {c.get('total_batches', 0)}")
        print(f"    Attempters:  {c.get('total_attempters', 0)}")
        print(f"    Confirmed:   {c.get('total_confirmed', 0)}")
        print(f"    Occupations: {', '.join(c.get('occupations', []))}")
        print(f"    Last update: {c.get('last_updated', 'never')}")

    def print_attempter_history(self, attempter):
        """Print formatted attempter history."""
        history = self.get_attempter_history(attempter)
        profile = history['profile']

        if not profile:
            print(f"\n  No history found for attempter: {attempter}")
            return

        p = profile
        print(f"\n{'='*70}")
        print(f"  ATTEMPTER HISTORY: {p['attempter']}")
        print(f"{'='*70}")
        print(f"    Risk tier:    {p.get('risk_tier', '?')}")
        print(f"    Submissions:  {p.get('total_submissions', 0)}")
        print(f"    Flag rate:    {p.get('flag_rate', 0):.1%}")
        print(f"    First seen:   {p.get('first_seen', '?')[:10]}")
        print(f"    Last seen:    {p.get('last_seen', '?')[:10]}")
        print(f"    Batches:      {len(p.get('batches', []))}")
        print(f"    Occupations:  {', '.join(p.get('occupations', []))}")

        dets = p.get('determinations', {})
        print(f"    Determinations: R={dets.get('RED', 0)} A={dets.get('AMBER', 0)} "
              f"Y={dets.get('YELLOW', 0)} G={dets.get('GREEN', 0)}")

        confirmed_ai = p.get('confirmed_ai', 0)
        confirmed_human = p.get('confirmed_human', 0)
        if confirmed_ai or confirmed_human:
            print(f"    Confirmed:    AI={confirmed_ai}  Human={confirmed_human}")

        if p.get('primary_detection_channel'):
            print(f"    Primary channel: {p['primary_detection_channel']}")

        subs = history['submissions']
        if subs:
            print(f"\n    Recent submissions ({len(subs)} total):")
            for s in subs[-5:]:
                print(f"      {s.get('task_id', '?')[:15]:15s} "
                      f"[{s.get('determination', '?')}] "
                      f"conf={s.get('confidence', 0):.2f} "
                      f"{s.get('occupation', '')[:25]}")
