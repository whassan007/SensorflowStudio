"""The eight advisory agents.

Each agent shares the BaseAgent contract (typed input/output, confidence,
failure handling, escalation rules, human-review triggers) and follows the
platform's copilot pattern: an OPTIONAL local LLM (Ollama) may add a
natural-language rationale, but every agent has a DETERMINISTIC RULE-BASED
core so the whole system runs and tests without any model. Agents are
structurally advisory (AgentResult.authority == "ADVISORY_ONLY"); the
deterministic policy engine in policy.py is the only launch authority.
"""

from sensorflow.agentic.agents.base import BaseAgent  # noqa: F401
from sensorflow.agentic.agents.failure_detection import FailureDetectionAgent  # noqa: F401
from sensorflow.agentic.agents.vlm_scene import VLMSceneAnalysisAgent  # noqa: F401
from sensorflow.agentic.agents.fusion_verification import SensorFusionVerificationAgent  # noqa: F401
from sensorflow.agentic.agents.scenario_mining import ScenarioMiningAgent  # noqa: F401
from sensorflow.agentic.agents.statistical import StatisticalRegressionAgent  # noqa: F401
from sensorflow.agentic.agents.safety_impact import SafetyImpactAgent  # noqa: F401
from sensorflow.agentic.agents.launch_decision import LaunchDecisionAgent  # noqa: F401
from sensorflow.agentic.agents.flywheel_agent import EvalFlywheelAgent  # noqa: F401
