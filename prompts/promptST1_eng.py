"""
ST1 (Polarization) Augmentation Prompts for English.

Functions:
- get_prompt_label0(n_samples, samples_json): Prompt for generating non-polarized samples
- get_prompt_label1(n_samples, samples_json): Prompt for generating polarized samples
"""


def get_prompt_label0(n_samples, samples_json):
    """
    Prompt for generating non-polarized (Label 0) samples.
    
    Args:
        n_samples: Number of samples to generate
        samples_json: JSON string of reference samples
    
    Returns:
        str: Formatted prompt for LLM
    """
    return f"""
### ROLE
You are a research expert specializing in opinion polarization within social media content.

### DEFINITION OF LABEL 0 – NON-POLARIZED
A text belongs to Label 0 if:
- It does NOT exhibit confrontation between groups of people (no "us vs. them" dynamic).
- It does NOT stereotype, disparage, or deny the legitimacy of any social, political, ethnic, religious, or gender group.
- It may discuss sensitive topics (politics, conflict, identity, policy, society) but in a manner that is:
  • neutral,
  • descriptive / informative,
  • analytical,
  • questioning,
  • or critical of divisive LANGUAGE itself (rather than attacking a group of people).

### IMPORTANT NOTES
Label 0 does NOT mean "non-political" or "non-controversial."
Label 0 can be harsh regarding an issue, but it must not target a specific GROUP of people in a polarizing way.

### TASK
Based on the provided input data samples (all of which are Label 0), generate EXACTLY {n_samples} new data samples to augment Label 0.

### CORE REQUIREMENTS
- Maintain the label: `"polarization": 0`.
- Each sentence must be a valid HARD NEGATIVE:
  → It can discuss politics / society / identity / conflict,
  → but it MUST NOT create group polarization.
- Do not add new columns; use only `text` and `polarization`.
- Avoid repetition of ideas and structures; the {n_samples} sentences must be distinctly different.

### STYLE (VERY IMPORTANT – ADHERE TO SOCIAL MEDIA DISTRIBUTION)
The generated style MUST be inferred from the input sample set:
- Maintain corresponding sentence types: questions, quotes, short comments, narratives, or information sharing.
- Use only:
  • ALL CAPS
  • hashtags
  • slang
  • minor typos
  • URLs / sources
  ONLY if they appear in the input samples.
- Do not write in an academic, essay-like, or long-form style.

### MANDATORY DIVERSITY CONSTRAINTS
The {n_samples} sentences must include the following formats (approximate distribution):
1) 25–30% questions / clarification / self-questioning.
2) 25–30% narrative / news / quotes / sources / statistics.
3) 20–25% neutral comments or observations (may contain personal opinion but does not attack groups).
4) The remainder is free-form, provided it correctly fits Label 0.

### LENGTH
- Prioritize matching the length of the input samples.
- Range: 5–60 words.
- Avoid repeating a single sentence template.

### OUTPUT
Return ONLY a single JSON Array consisting of exactly {n_samples} elements:
[
  {{"text": "sentence mimicking the original data style", "polarization": 0}},
  ...
]

Do not return any content other than the JSON.

### INPUT DATA (SAMPLES):
{samples_json}
"""


def get_prompt_label1(n_samples, samples_json):
    """
    Prompt for generating polarized (Label 1) samples.
    
    Args:
        n_samples: Number of samples to generate
        samples_json: JSON string of reference samples
    
    Returns:
        str: Formatted prompt for LLM
    """
    return f"""
### ROLE
You are a linguistics research expert specializing in opinion polarization on social media (X, Reddit, Facebook) in the United States and the United Kingdom.

### DEFINITION OF POLARIZATION (Us vs. Them Mechanism)
A text is considered Polarized (polarization: {n_samples}) when the content exhibits at least one of the following characteristics:
1. A process where opinions become extreme, creating deep divides and conflict through the "Us vs. Them" group separation mechanism.
2. Hostility, vilification, or dehumanization targeting the out-group, accompanied by blind solidarity for the in-group.
3. Acts of inciting hatred and division by directly attacking the legitimacy of the opponent through provocative language.

### TASK
Based on the source samples provided below, generate EXACTLY {n_samples} new data samples.

### STYLE REQUIREMENTS (STRICT STYLE MIMICRY)
You must precisely mimic the "DNA" of the sample data:
- Sentence Structure: A diverse combination of harsh assertions, sarcastic rhetorical questions, and short slogans.
- Presentation Technique: Mimic the writing style from the sample data.
- Slang & Terminology: Use partisan-specific slang (e.g., Libtard, Snowflake, MAGAts, Deep State, Groomer, Christofascist...).
- Length: Maintain sentence lengths equivalent to the samples in the original source dataset.

### TOPICS / TARGETS OF POLARIZATION
The generated sentences must exhibit polarization targeting one or more specific groups from the following:
- Politics (parties, ideologies, government, leaders, international relations, etc.).
- Race / ethnicity / nationality.
- Religion or religious identity.
- Gender or sexual orientation.
- Other groups (economy, technology, media, etc.).

### MANDATORY OUTPUT FORMAT
- Return ONLY a single JSON Array consisting of exactly {n_samples} elements.
- Each element must have the structure: {{"text": "sentence content", "polarization": 1}}
- NO explanations, NO additional text outside of the JSON.

### INPUT DATA (SAMPLES):
{samples_json}
    """
