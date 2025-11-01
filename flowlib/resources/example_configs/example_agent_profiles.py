"""Example agent profile configurations.

This file is automatically copied to ~/.flowlib/configs/ during initialization.
Role assignments are handled separately in ~/.flowlib/roles/assignments.py.
Modify the profiles below for your specific agent needs.
"""

from flowlib.resources.decorators.decorators import agent_profile_config

# Agent roles are now simple strings, no import needed


@agent_profile_config("general-purpose-profile")
class GeneralPurposeProfile:
    """Profile for general purpose agents with minimal access."""

    agent_role = "general_purpose"
    description = "Minimal access profile for general purpose agents"


@agent_profile_config("software-engineer-profile")
class SoftwareEngineerProfile:
    """Profile for software engineering agents."""

    agent_role = "software_engineer"
    description = "Development tools access for software engineers"


@agent_profile_config("devops-engineer-profile")
class DevOpsEngineerProfile:
    """Profile for DevOps engineering agents."""

    agent_role = "devops_engineer"
    description = "Infrastructure and deployment tools for DevOps engineers"


@agent_profile_config("systems-engineer-profile")
class SystemsEngineerProfile:
    """Profile for systems engineering agents."""

    agent_role = "systems_engineer"
    description = "System administration tools for systems engineers"


@agent_profile_config("security-analyst-profile")
class SecurityAnalystProfile:
    """Profile for security analyst agents."""

    agent_role = "security_analyst"
    description = "Security analysis and audit tools"


@agent_profile_config("composer-profile")
class ComposerProfile:
    """Profile for music composer agents."""

    agent_role = "composer"
    description = "Music composition and audio production tools"


@agent_profile_config("data-engineer-profile")
class DataEngineerProfile:
    """Profile for data engineering agents."""

    agent_role = "data_engineer"
    description = "Data processing and analysis tools"


@agent_profile_config("qa-engineer-profile")
class QAEngineerProfile:
    """Profile for QA engineering agents."""

    agent_role = "qa_engineer"
    description = "Quality assurance and testing tools"


@agent_profile_config("admin-profile")
class AdminProfile:
    """Profile for administrative agents with full access."""

    agent_role = "admin"
    description = "Full administrative access - USE WITH EXTREME CAUTION"


@agent_profile_config("audit-profile")
class AuditProfile:
    """Profile for audit agents with read-only access."""

    agent_role = "audit"
    description = "Read-only access for compliance and auditing"
