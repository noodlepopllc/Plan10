import json, sys, argparse, random
sys.stdout.reconfigure(encoding='utf-8')

from plan10.lib.qwen_llm import llm_analyze_media
from pathlib import Path

# ============================================================================
# DEFAULT SEED (used when no seed file provided)
# ============================================================================

SEED = """
Characters:
- Maria: anxious, values honesty
- Lisa: private, texting someone

Location: dorm room

Motivations:
- Maria wants openness.
- Lisa wants privacy.

Spark:
Maria thinks Lisa is hiding something.

Story Goal:
Maria wants honesty, but Lisa appears to be hiding something.

Initial Situation:
Lisa is texting on her bed. Maria enters the room.
"""

# ============================================================================
# TOPICAL SEED GENERATOR (for when topic is provided but no seed file)
# ============================================================================

TOPICAL_SCENARIOS = ['DEBATE', 'TEACHING', 'DISCUSSION', 'MENTORSHIP']

TOPICAL_SEED_GENERATOR_TEMPLATE = '''
🎓 TOPICAL SCENE SEED GENERATOR (EDUCATIONAL/DEBATE)
ROLE — Generate structured seeds for educational or debate scenes

⭐ SCENARIO TYPE: {scenario}
⭐ TOPIC: {topic}

⭐ CHARACTER ROLES

For DEBATE scenarios:
- Character A: Holds position X, uses logic/evidence type Y
- Character B: Holds opposing position, uses different reasoning style
- Both must be articulate, passionate, and have distinct argumentation styles

For TEACHING scenarios:
- Teacher: Expert with deep knowledge, patient but demanding
- Student: Eager but struggling, asks good questions, makes mistakes
- Dynamic: Knowledge transfer with friction/learning curve

For DISCUSSION scenarios:
- Character A: Brings perspective/experience X
- Character B: Brings different perspective/experience Y
- Both learn from each other, reach synthesis

For MENTORSHIP scenarios:
- Mentor: Wise, experienced, uses Socratic method
- Protégé: Talented but raw, needs guidance
- Dynamic: Growth through challenge and reflection

⭐ SEED STRUCTURE (OUTPUT EXACTLY THIS FORMAT)

**Scenario Type**: [DEBATE/TEACHING/DISCUSSION/MENTORSHIP]
**Topic**: [specific topic with clear scope]
**Core Question**: [the central question being explored/argued]

**Characters** (2 characters):
- [Name]: [age], [gender], [race/species if relevant], [physical description: build, face, distinctive features, FULL clothing with material/color/condition, hair style/color/length, footwear]. [Role in scenario: their position/expertise]. [Argumentation/teaching style: how they communicate their ideas].
- [Name]: [same structure]. [Opposing role/learning position]. [Communication style].

**Location**:
[Name of location]. [2-3 sentences describing space: size, architectural features, lighting, textures, sounds, atmosphere]. [3-5 specific objects relevant to the topic: books, tools, artifacts, technology, etc.]. [What makes this location appropriate for this type of discussion].

**Story Spark**:
[1-2 sentences describing what initiates the debate/discussion. Must be concrete: a question asked, a challenge issued, a problem presented, a disagreement revealed].

**Character Goals**:
- [Character A]: [Specific goal: convince the other, teach a concept, solve a problem, reach understanding]. Must be achievable within the scene.
- [Character B]: [Specific goal: defend position, learn the skill, challenge assumptions, find common ground]. Should create tension with Character A's goal.

**Key Arguments/Teaching Points** (3-5 points that will emerge):
1. [First major point/argument that will be raised]
2. [Second point that builds on or challenges the first]
3. [Third point that escalates the discussion]
4. [Optional: fourth point if needed for complexity]
5. [Optional: resolution/synthesis point]

**Initial Situation**:
[2-3 sentences describing exactly where each character is positioned, what they're doing with their hands/body, and the immediate context. Must be concrete and filmable.]

⭐ QUALITY GUARDRAILS
- Topic must be specific enough to debate/teach in 20-40 beats
- Characters must have distinct communication styles (not just different opinions)
- Location must contain objects relevant to the topic (props for demonstration, reference materials, tools)
- Goals must be in tension but both achievable
- Initial situation must show characters in positions that reflect their roles
- Dialog will drive 70%+ of the scene, so characters must be articulate

⭐ BEGIN OUTPUT NOW
Generate one complete seed in the exact format above. No commentary or explanation.
'''

# ============================================================================
# ACTION SCENE EXPANSION PROMPT
# ============================================================================

ALTERNATIVE = '''
🎬 SCENE EXPANSION PROMPT
Using the structured seed below, write a single continuous scene consisting of:
1 COLD OPEN (establishing moment, not counted as a scene beat)
8–12 SCENE BEATS (action/dialog within this single moment)

------------------------------------------------------------
COLD OPEN (MANDATORY — DOES NOT COUNT TOWARD BEAT TOTAL)
------------------------------------------------------------

The cold open is a rich prose snapshot of the initial situation from the seed.
Write it as literary prose — atmospheric, detailed, story-like.

REQUIRED ELEMENTS:
- Environment: lighting, textures, atmosphere, time of day
- Each character: species/ethnicity (if in seed), build, face, clothing, hair, current position
- What each character is caught in the middle of doing (from seed's "Initial Situation")

FORBIDDEN:
- Any action that advances the plot
- Characters moving toward goals
- Conflict escalation
- Dialog beyond ambient/greeting

Think of it as: "Characters caught mid-state" → then the story begins.

CRITICAL: After the cold open, output this exact marker on its own line:
******* COLD OPEN END ****

------------------------------------------------------------
STORY BEATS (8–12 TOTAL)
------------------------------------------------------------

Each beat advances tension through action or dialog. Escalate conflict based on the story spark.

BEAT STRUCTURE:
- 1-2 active characters performing visible actions
- Dialog appears every 2-3 beats
- Dialog must advance story (goals, conflict, commands)
- Beat ends when primary character goal is resolved or definitively failed

ENRICHED MACRO ACTIONS:
Each action beat should be a clear, visible movement WITH descriptive context.
Structure: [verb] + [object] + [quality/manner]
Examples:
- "grasps cylinder" → "grasps heavy metal cylinder with both hands"
- "steps forward" → "steps forward with determined stride"

Rules:
- Include physical qualities: weight, texture, temperature
- Include manner: speed, force, direction
- Keep actions macro-level (visible body movements, not subtle gestures)

------------------------------------------------------------
DIALOG RULES
------------------------------------------------------------

Dialog must appear every 2-3 beats.
Dialog must advance the story (state goals, create conflict, give commands).
Every dialog line must be paired with a physical action.

Characters may REFERENCE past events in dialog, but they must NOT tell full stories within the scene.
❌ FORBIDDEN: "Let me tell you about the time I fought bandits. It started when I was walking through the Darkwood Forest..."
✔ ALLOWED: "I fought bandits last week. Nasty business."

------------------------------------------------------------
PHYSICAL GROUNDING
------------------------------------------------------------

Keep the scene physically anchored in location.
Express internal states strictly through observable physical behavior.

------------------------------------------------------------
OUTPUT FORMAT
------------------------------------------------------------

Write in standard literary prose. Begin with the cold open (static snapshot only), insert the ******* COLD OPEN END **** marker, then write exactly 8-12 story beats. Dialog must appear every 2-3 beats.
'''

# ============================================================================
# TOPICAL SCENE EXPANSION PROMPT
# ============================================================================

TOPICAL_EXPANDER = '''
🎓 TOPICAL SCENE EXPANDER (EDUCATIONAL/DEBATE)

Using the structured seed below, write a coherent topical scene consisting of:
1 COLD OPEN (establishing moment, not counted as a scene beat)
20-40 SCENE BEATS (dialog-heavy intellectual exchange)

CORE DIRECTIVES:

COLD OPEN (MANDATORY — DOES NOT COUNT TOWARD BEAT TOTAL)
Write a rich, detailed establishing sequence BEFORE the scene begins. Include:
- Environment (2-3 sentences): Lighting, textures, sounds, spatial layout, atmosphere
- Each Character (2-3 sentences per character): Build, face, distinctive features, complete clothing (head-to-toe), hair, current position, body state, and what they're holding/interacting with (books, tools, props relevant to topic)

❌ FORBIDDEN IN COLD OPEN: Backstory, internal thoughts, future actions, dialog.

⭐ SPECIES/ETHNICITY MUST BE EXPLICIT
If the seed specifies a species (elf, half-elf, android, werecreature, alien, etc.), you MUST explicitly state it in the cold open.

CRITICAL MARKER: After the cold open paragraphs, you MUST output this exact marker on its own line:
******* COLD OPEN END ****

⭐ SCENE STRUCTURE FOR TOPICAL CONTENT

For DEBATE scenarios:
- Beats 1-3: Opening statements, establishing positions
- Beats 4-15: First round of arguments (3-4 major points with responses)
- Beats 16-25: Second round (counter-arguments, examples, evidence)
- Beats 26-35: Final round (escalation, emotional appeals, core values)
- Beats 36-40: Resolution or acknowledged impasse

For TEACHING scenarios:
- Beats 1-3: Introduction of concept, initial explanation
- Beats 4-15: Core teaching (break concept into 3-4 sub-points, each with explanation + student questions)
- Beats 16-25: Practice/application (student tries, makes mistakes, teacher corrects)
- Beats 26-35: Advanced concepts or troubleshooting common errors
- Beats 36-40: Summary, check understanding, next steps

For DISCUSSION scenarios:
- Beats 1-5: Both perspectives introduced
- Beats 6-20: Exploration of each perspective with examples
- Beats 21-30: Finding common ground or identifying irreconcilable differences
- Beats 31-40: Synthesis or agreed disagreement

For MENTORSHIP scenarios:
- Beats 1-5: Mentor assesses current state, identifies challenge
- Beats 6-20: Guided problem-solving (mentor asks questions, protégé attempts solutions)
- Beats 21-30: Breakthrough moment or realization
- Beats 31-40: Reflection on learning, application to future

⭐ DIALOG DENSITY (CRITICAL)
- 70-80% of beats must contain dialog
- Dialog lines should be 8-20 words (substantive but not monologues)
- Every dialog beat must include physical action (gesturing, writing, demonstrating, reacting)
- Characters should reference objects in the environment (pointing to books, picking up tools, writing on surfaces)

⭐ INTELLECTUAL PROGRESSION
Each beat must advance the intellectual exchange:
- Introduce a new point/argument
- Respond to previous point
- Provide example or evidence
- Ask clarifying question
- Show emotional reaction to idea (frustration, excitement, confusion, realization)

❌ FORBIDDEN: Repeating the same argument in different words
✅ REQUIRED: Each beat adds new information, perspective, or development

⭐ PHYSICAL ANCHORING
Even in dialog-heavy scenes, characters must:
- Move through space (pacing, approaching, retreating)
- Interact with props (picking up books, writing on boards, demonstrating with objects)
- Show emotional states through body language (leaning forward in excitement, crossing arms in defensiveness, rubbing temples in frustration)
- Use gestures to emphasize points (pointing, counting on fingers, spreading hands)

⭐ ENRICHED MACRO ACTIONS
Each action beat should be a clear, visible movement WITH descriptive context.

Structure: [verb] + [object] + [quality/manner]

Examples:
- "points to equation" → "points to complex equation on whiteboard with marker"
- "opens book" → "opens leather-bound book to marked page"
- "gestures emphatically" → "gestures emphatically with both hands, palms up"

Rules:
- Use 12-15 words for action descriptions
- Include physical qualities: weight, texture, material
- Include manner: speed, force, direction
- Keep actions macro-level (visible body movements, not subtle gestures)

⭐ EMOTIONAL PHYSICALITY
Show intellectual engagement through body language:
- Excitement: leaning forward, eyes wide, rapid gestures
- Frustration: rubbing temples, pacing, slamming hand on table
- Confusion: tilting head, furrowed brow, hesitant gestures
- Realization: eyes widening, sudden stillness, pointing emphatically
- Defensiveness: crossed arms, leaning back, narrowed eyes
- Agreement: nodding, open palms, relaxed posture

HARD BEAT COUNT: 20-40 BEATS
After the cold open, write 20-40 beats depending on topic complexity.
- Simple topic with 2 characters: 20-25 beats
- Complex topic requiring multiple points: 30-35 beats
- Topic with demonstration/practice: 35-40 beats

Stop when the intellectual goal is achieved or acknowledged as unachievable.

OUTPUT FORMAT:
Write in standard literary prose. Begin with the cold open, insert the ******* COLD OPEN END **** marker, then write 20-40 numbered beats. Each beat is ONE paragraph with dialog and/or action. No nested stories.

BEGIN OUTPUT NOW
'''

TOPICAL_SEED_GENERATOR_TEMPLATE = '''
🎓 TOPICAL SCENE SEED GENERATOR (EDUCATIONAL/DEBATE)
ROLE — Generate structured seeds for educational or debate scenes

⭐ SCENARIO TYPE: {scenario}
⭐ TOPIC: {topic}

⭐ CHARACTER ROLES
{seed}
'''

def topical_seed_generator(scenario, topic, seed):
    return TOPICAL_SEED_GENERATOR_TEMPLATE.format(
        scenario=scenario,
        topic=topic, 
        seed=seed
    )

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def run_prompt(prompt, system, pth, max_tokens=8192):
    if not Path(pth).exists():
        result = llm_analyze_media(
            media="",
            prompt=prompt,
            system=system,
            max_tokens=max_tokens
        )['analysis']

        Path(pth).write_text(result, encoding='utf-8')
        print(f'Wrote {pth}')
        return result

    else:
        print(f'{pth} Exists')
        return Path(pth).read_text(encoding='utf-8')


# ============================================================================
# MAIN
# ============================================================================
TOPICS = [
    'Philosophy/Ethics', 'Science/Technology', 'History/Politics',
    'Art/Culture', 'Personal Development', 'Social Issues',
    'Creative Process', 'Professional Skills'
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Expand a seed into a full scene')
    parser.add_argument('-S', '--seed', type=str, default=None)
    parser.add_argument('-O', '--output', type=str, default='story.txt')
    parser.add_argument('-T', '--topic', type=str, default=None)
    parser.add_argument('--scenario', type=str, default=None)
    parser.add_argument('--topical', action='store_true')
    parser.add_argument('--max-tokens', type=int, default=None,
                        help='Override default token budget')
    args = parser.parse_args()

    output_path = Path(args.output)
    seed_text = ''

    # Set token budget based on expander type
    if args.max_tokens:
        token_budget = args.max_tokens
    elif args.topical:
        token_budget = 16384  # Topical needs more room for 20-40 beats
    else:
        token_budget = 8192   # Action scenes need less

    if args.seed:
        seed_text = Path(args.seed).read_text(encoding='utf-8')
        print(f'Using seed from {args.seed}')

    if args.topical:
        scenario = args.scenario.upper() if args.scenario and args.scenario.upper() in TOPICAL_SCENARIOS else random.choice(TOPICAL_SCENARIOS)
        topic = args.topic if args.topic else random.choice(TOPICS)

        seed_path = output_path.with_name(output_path.stem + '_seed.txt')
        inputs = f'Generate a topical seed\nTopic: {topic}\nScenario Type: {scenario}\n'
        seed_text = run_prompt(inputs, topical_seed_generator(scenario, topic, seed_text), str(seed_path), max_tokens=4096)
        print(f'Generated topical seed: {scenario} / {topic}')

    elif not seed_text:
        seed_text = SEED 
        print('Using default seed')

    expander = TOPICAL_EXPANDER if args.topical else ALTERNATIVE
    expander_type = 'topical' if args.topical else 'action'
    print(f'Expanding with {expander_type} expander (token budget: {token_budget})')

    print(run_prompt(seed_text, expander, str(output_path), max_tokens=token_budget))
