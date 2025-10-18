"""Example agent profile configurations.

This file is automatically copied to ~/.flowlib/configs/ during initialization.
Role assignments are handled separately in ~/.flowlib/roles/assignments.py.
Modify the profiles below for your specific agent needs.
"""

from typing import Any

from flowlib.resources.decorators.decorators import agent_profile_config
from flowlib.resources.models.config_resource import AgentProfileConfigResource

# Agent roles are now simple strings, no import needed


@agent_profile_config("general-purpose-profile")
class GeneralPurposeProfile(AgentProfileConfigResource):
    """Profile for general purpose agents with minimal access."""

    def __init__(self, name: str, type: str, **kwargs: Any) -> None:
        super().__init__(
            name=name,
            type=type,
            agent_role="general_purpose",
            description="Minimal access profile for general purpose agents"
        )


@agent_profile_config("software-engineer-profile")
class SoftwareEngineerProfile(AgentProfileConfigResource):
    """Profile for software engineering agents."""

    def __init__(self, name: str, type: str, **kwargs: Any) -> None:
        super().__init__(
            name=name,
            type=type,
            agent_role="software_engineer",
            description="Development tools access for software engineers"
        )


@agent_profile_config("devops-engineer-profile")
class DevOpsEngineerProfile(AgentProfileConfigResource):
    """Profile for DevOps engineering agents."""

    def __init__(self, name: str, type: str, **kwargs: Any) -> None:
        super().__init__(
            name=name,
            type=type,
            agent_role="devops_engineer",
            description="Infrastructure and deployment tools for DevOps engineers"
        )


@agent_profile_config("systems-engineer-profile")
class SystemsEngineerProfile(AgentProfileConfigResource):
    """Profile for systems engineering agents."""

    def __init__(self, name: str, type: str, **kwargs: Any) -> None:
        super().__init__(
            name=name,
            type=type,
            agent_role="systems_engineer",
            description="System administration tools for systems engineers"
        )


@agent_profile_config("security-analyst-profile")
class SecurityAnalystProfile(AgentProfileConfigResource):
    """Profile for security analyst agents."""

    def __init__(self, name: str, type: str, **kwargs: Any) -> None:
        super().__init__(
            name=name,
            type=type,
            agent_role="security_analyst",
            description="Security analysis and audit tools"
        )


@agent_profile_config("composer-profile")
class ComposerProfile(AgentProfileConfigResource):
    """Profile for music composer agents."""

    def __init__(self, name: str, type: str, **kwargs: Any) -> None:
        super().__init__(
            name=name,
            type=type,
            agent_role="composer",
            description="Music composition and audio production tools"
        )


@agent_profile_config("data-engineer-profile")
class DataEngineerProfile(AgentProfileConfigResource):
    """Profile for data engineering agents."""

    def __init__(self, name: str, type: str, **kwargs: Any) -> None:
        super().__init__(
            name=name,
            type=type,
            agent_role="data_engineer",
            description="Data processing and analysis tools"
        )


@agent_profile_config("qa-engineer-profile")
class QAEngineerProfile(AgentProfileConfigResource):
    """Profile for QA engineering agents."""

    def __init__(self, name: str, type: str, **kwargs: Any) -> None:
        super().__init__(
            name=name,
            type=type,
            agent_role="qa_engineer",
            description="Quality assurance and testing tools"
        )


@agent_profile_config("admin-profile")
class AdminProfile(AgentProfileConfigResource):
    """Profile for administrative agents with full access."""

    def __init__(self, name: str, type: str, **kwargs: Any) -> None:
        super().__init__(
            name=name,
            type=type,
            agent_role="admin",
            description="Full administrative access - USE WITH EXTREME CAUTION"
        )


@agent_profile_config("audit-profile")
class AuditProfile(AgentProfileConfigResource):
    """Profile for audit agents with read-only access."""

    def __init__(self, name: str, type: str, **kwargs: Any) -> None:
        super().__init__(
            name=name,
            type=type,
            agent_role="audit",
            description="Read-only access for compliance and auditing"
        )
