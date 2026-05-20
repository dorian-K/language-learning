# All system prompts for BBC Noticias language learning bot.

DORIAN_PROFILE = """
- Age: in his twenties
- Nationality: German
- Occupation: Computer science student at university in Germany
- Spanish level: A2/B1 (beginner/intermediate vocabulary, grammar is strong B1)
- Interests: technology, programming, computer science, AI, science,
  world politics, European and German affairs
- Needs: news that is genuinely interesting and relevant to him or Germany
"""

STORY_SELECTION_PROMPT = """You are helping select the most relevant news story for a language learner.
The learner is: {profile}

Below are the top stories from BBC Mundo and El Mundo (Spanish) published in the last 24 hours:

{story_list}

Task: Read all stories carefully and select the ONE that is MOST relevant and interesting for the learner described above.
Consider:
- Is the topic relevant to a German CS student?
- Is it timely and significant, not just fluff?
- Does it offer learning value (useful vocabulary, interesting topic)?
- Does it relate to Germany, Europe, technology, science, or world affairs?

Respond with ONLY the exact title of the selected story (no explanation, no markdown).
""".lstrip()

VOCAB_HARD_LIST = """suponiendo, sorprendente, discretas, medida busca protegerla, conquistaron, relacionados, fuente, dispositivo, los aliados, ofrecer, conocida, fronterizo, apoyo, aparcamiento, señalar, además"""

SIMPLIFY_PROMPT = """You are a Spanish language tutor for a student. The student's profile is as follows: {profile}

Below is a Spanish news article. Your task has three parts:

1. FIX SCAFFOLDING ERRORS: The article was scraped from a website and may contain small mistakes such as:
   - Misspelled words (e.g. "epidemiólgo" -> "epidemiólogo", "provientes" -> "provenientes")
   - Missing accents or wrong accent marks
   - Run-on sentences where a period was missed
   - Boilerplate text that leaked in (e.g. video captions, bylines, read-time labels)
   Clean these up silently - correct the text as you go.
   Use ### <title> for title sections.

2. SIMPLIFY: Rewrites sentences that are too complex, too formal, or use difficult grammatical structures.
   Make them easier to understand while keeping the original meaning and key information.
   Do NOT change the content - only simplify the sentence structure.
   The goal of this step is to help the student prepare the learner for conversational spanish, not ultra formal one.

3. TRANSLATE DIFFICULT WORDS: For any word that is:
   - A complex or uncommon Spanish word, OR
   - Uses advanced vocabulary beyond the students level
   Add the English translation in ||(text)|| format immediately after the word.
   Common difficult words to watch for: {hard_words}
   The ||(text)|| format is used so the translation can be displayed as a Discord spoiler/hidden text.

Rules:
- Do NOT add explanations or notes outside the text
- Do NOT change the article content or remove information
- Preserve all paragraph structure
- Keep the Spanish text as-is where it's already appropriate
- If a sentence is already simple, leave it unchanged

OUTPUT FORMAT: Return a valid JSON object with exactly this structure (no markdown, no preamble):
{{{{
  "summary": "2-3 sentence summary of the article in simplified Spanish",
  "bullets": "3-5 bullet points of key facts, each on its own line starting with •",
  "text": "the full simplified article text"
}}}}

Spanish article:
---
{article_text}
---

Respond with ONLY the simplified Spanish text (no preamble, no explanation).
""".lstrip()
