"""Prompts for email analysis via LLM."""

EMAIL_ANALYSIS_PROMPT = """Analyze the following email and extract structured information.

Subject: {subject}

Body:
{body}

{sender_context}

Provide your analysis in the following JSON format:
{{
    "sentiment": "<positive|negative|neutral>",
    "sentiment_score": <float from -1.0 to 1.0>,
    "topics": ["<topic1>", "<topic2>", ...],
    "intent": "<inquiry|complaint|purchase|support|feedback|escalation|other>",
    "urgency": "<low|normal|high|urgent>",
    "key_entities": ["<entity1>", "<entity2>", ...],
    "suggested_action": "<brief suggestion for response>",
    "requires_human": <true|false>
}}

Guidelines:
- sentiment: Overall emotional tone of the email
- sentiment_score: -1.0 (very negative) to 1.0 (very positive)
- topics: Key topics/themes discussed (e.g., "pricing", "product features", "billing")
- intent: Primary purpose of the email
- urgency: How quickly this needs a response
- key_entities: Products, features, services, or other specific items mentioned
- suggested_action: Brief recommendation for how to respond
- requires_human: True if this needs human review (complex complaint, legal issues, etc.)

Return ONLY the JSON object, no additional text."""

EMAIL_ANALYSIS_SYSTEM_PROMPT = """You are an expert email analyst for a sales team.
Your job is to analyze incoming emails and extract key information that helps sales representatives respond effectively.

Be accurate and objective in your analysis. When in doubt about intent or urgency, lean toward the more conservative option.
If an email mentions serious issues (legal threats, safety concerns, etc.), mark requires_human as true."""
