"""Shared constants used by multiple modules (CLI, GUI, dashboard)."""

PIPELINE_VERSION = 'v0.68'

GROUND_TRUTH_LABELS = ['ai', 'human', 'unsure']
STREAMLIT_MIN_VERSION = 'streamlit>=1.20'

DETERMINATION_ICONS = {
    'RED': '\U0001f534', 'AMBER': '\U0001f7e0', 'YELLOW': '\U0001f7e1',
    'GREEN': '\U0001f7e2', 'MIXED': '\U0001f535', 'REVIEW': '\u26aa',
}

FLAGGED_DETERMINATIONS = frozenset({'RED', 'AMBER', 'MIXED'})


def get_length_bin(word_count):
    """Classify word count into a length bin for stratified calibration."""
    if word_count < 100:
        return 'short'
    elif word_count < 300:
        return 'medium'
    elif word_count < 800:
        return 'long'
    return 'very_long'


def is_ssi_triggered(voice_dis, word_count):
    """Check if the Sterile Specification Index (SSI) is triggered."""
    threshold = 5.0 if voice_dis.get('contractions', 0) == 0 else 7.0
    return (
        voice_dis.get('spec_score', 0) >= threshold
        and voice_dis.get('voice_score', 1.0) < 0.5
        and voice_dis.get('hedges', 1) == 0
        and word_count >= 150
    )
