"""
skills/__init__.py -- Skill registry for the autonomous agent.

Skills are modular intelligence units that provide pre-action reasoning,
validation, and classification. They run BEFORE the agent commits to any
action, preventing blind behavior.
"""

from skills.contact_classifier import ContactClassifier
from skills.message_guard import MessageGuard
from skills.context_analyzer import ContextAnalyzer
from skills.silence_policy import SilencePolicy
from skills.human_mimicry import HumanMimicry

__all__ = [
    "ContactClassifier",
    "MessageGuard",
    "ContextAnalyzer",
    "SilencePolicy",
    "HumanMimicry",
]
