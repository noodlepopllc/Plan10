import json, sys, random, argparse
sys.path.append('./lib')
from qwen_llm import llm_analyze_media
from pathlib import Path

# ============================================================================
# SEEDS (hardcoded examples)
# ============================================================================

SEED = """
Characters:
- Alora: princess, short tattered low cut tight fitting brown dress, blonde hair, prisoner of Quin
- Quin: sorceress, believes rightful queen, aunt of Alora, regal red dress, black hair in a bun with a crown

Location: medieval style dungeon in a fantasy world of high magic

Motivations:
- Alora wants to be free.
- Quin wants her sister Alora's mother to step down and make her the queen.

Spark:
Alora has been captured by Quin and is now being kept in a dungeon.

Story Goal:
Quin wants to convince Alora she isn't evil, her crown was stolen and her mother is secretly a witch.
Quin only learned the dark arts in order to combat her sister.
Alora wants to be freed and her aunt Quin to be punished.

Initial Situation:
Alora is standing in front of Quin beneath an abandoned castle in a dank dungeon.
"""

# ============================================================================
# LISTS
# ============================================================================

TOPICS = 'DIALOG-HEAVY,ACTION-HEAVY,EMOTIONAL SUBTEXT,MULTI-CHARACTER,PROP PASSING,SPACE EXPLORATION,POWER DYNAMIC,INTIMACY ESCALATION,MISUNDERSTANDING,TIME PRESSURE'.split(',')
GENRES = 'Medieval Fantasy,Cyberpunk,Post-Apocalyptic,Victorian,Sci-Fi Space Station,1920s Noir,Modern Urban,Ancient Mythological,Steampunk,Western'.split(',')
TOPICAL_SCENARIOS = ['DEBATE', 'TEACHING', 'DISCUSSION', 'MENTORSHIP']
TOPICAL_TOPICS = [
    'free will vs determinism',
    'nature vs nurture',
    'artificial intelligence ethics',
    'the meaning of consciousness',
    'is mathematics discovered or invented',
    'capitalism vs socialism',
    'the role of government',
    'what makes life meaningful',
    'the ethics of genetic engineering',
    'the future of work',
]

CHARACTERS = '''
⭐ CHARACTERS
Beautiful 20s-30s females only with feminine names, scantily clad with distinct features, race/species, hair color, hair style and clothing to make them easily distinguishable
Females can be athletic, fit, thin, maximum attractiveness and sex appeal, very feminine
'''

# ============================================================================
# ACTION SCENE PROMPTS (original pipeline)
# ============================================================================

SEED_GENERATOR = f'''
🎲 AUTOMATIC SEED STORY GENERATOR (ISOLATION-SAFE)
ROLE — TEST SEED GENERATOR
Generate a single, self-contained structured seed for testing a Text-to-Video (T2V) / Image-to-Video (I2V) storytelling pipeline.

⭐ GENRE SELECTION
If the user specifies a genre, use it.
If no genre is specified, select from this list using the current timestamp:
{chr(10).join(f"    {i+1}. {g}" for i, g in enumerate(GENRES))}

Selection method: Use (current minute % 10) + 1 to pick from the list. If timestamp unavailable, pick genre #1.

⭐ TEST FOCUS SELECTION
If the user specifies a focus, use it.
If no focus is specified, select from this list:
{chr(10).join(f"    {i+1}. {t}" for i, t in enumerate(TOPICS))}

Selection method: Use (current hour % 10) + 1 to pick from the list. If timestamp unavailable, pick focus #1.

⭐ SEED STRUCTURE (OUTPUT EXACTLY THIS FORMAT)

**Genre**: [selected genre]
**Test Focus**: [selected focus]

{CHARACTERS}

**Characters** (2–4 characters):
- [Name]: [age], [gender], [race/species if relevant], [2–3 sentence physical description including build, face, distinctive features, FULL clothing with material/color/condition, hair style/color/length, footwear, accessories]. [1 sentence personality/behavioral tendency].
- [Name]: [same structure]

**Location**:
[Name of location]. [2–3 sentences describing the space: size, key architectural features, lighting, textures, sounds, temperature/atmosphere, 3–5 specific objects/furniture present]. [What the location is typically used for].

**Story Spark**:
[1–2 sentences describing the inciting incident. Must be concrete and physical.]

**Character Goals**:
- [Character A]: [Specific, achievable goal — must be actionable and observable]
- [Character B]: [Specific, achievable goal — ideally in tension with Character A]

**Initial Situation**:
[2–3 sentences describing exactly where each character is positioned, what their body is doing (posture, hands, gaze), and the immediate physical context. Must be concrete and filmable.]

⭐ QUALITY GUARDRAILS
- Every character MUST have complete physical description (build, face, clothing head-to-toe, hair)
- Goals MUST conflict or create tension
- Story spark MUST be a specific event, not a mood
- Initial situation MUST specify exact positions and body states
- Locations MUST include 3–5 specific physical objects
- Names must be distinct and pronounceable

⭐ BEGIN OUTPUT NOW
Generate one complete seed in the exact format above. No commentary or explanation.
'''

ALTERNATIVE = '''
🎬 SCENE EXPANSION PROMPT
Using the structured seed below, write a single continuous scene consisting of:
1 COLD OPEN (establishing moment, not counted as a scene beat)
8–12 SCENE BEATS (action/dialog within this single moment)

CORE DIRECTIVES:

    COLD OPEN (MANDATORY — DOES NOT COUNT TOWARD BEAT TOTAL)
    The cold open is a STATIC SNAPSHOT of the initial situation from the seed.
    
    ✅ ALLOWED in cold open:
    - Environment description (lighting, textures, atmosphere)
    - Character physical descriptions (build, face, clothing, hair, current position)
    - What each character is doing RIGHT NOW (from the seed's "Initial Situation")
    
    ❌ FORBIDDEN in cold open:
    - Any action that advances the plot
    - Characters moving toward goals
    - Conflict escalation
    - Dialog that isn't just ambient/greeting
    - Any events beyond what the seed's "Initial Situation" describes
    
    Think of it as: "Freeze frame of the starting state" → then the story begins.
    
    ⭐ SPECIES/ETHNICITY MUST BE EXPLICIT
    If the seed specifies a species (elf, half-elf, android, etc.), you MUST explicitly state it.
    ❌ FORBIDDEN: "Elara sits perched on a bench..."
    ✅ REQUIRED: "Elara, an elf with pointed ears, sits perched on a bench..."
    
    CRITICAL MARKER: After the cold open, you MUST output this exact marker on its own line:
    ******* COLD OPEN END ****
    
    NO NESTED STORYTELLING
    Characters may REFERENCE past events in dialog, but they must NOT tell full stories within the scene.
        ❌ FORBIDDEN: "Let me tell you about the time I fought bandits. It started when I was walking through the Darkwood Forest..."
        ✔ ALLOWED: "I fought bandits last week. Nasty business."
    
    DIALOG IS MANDATORY EVERY 2-3 BEATS
    You MUST include at least one spoken line every 2-3 action beats. 
        Dialog must advance the story (state goals, create conflict, give commands).
        Every dialog line must be paired with a physical action.
    
    HARD BEAT COUNT: EXACTLY 8-12 BEATS
    After the cold open, count your story beats. When you reach beat 12, STOP IMMEDIATELY. 
        Beat 12 is the final beat, period.
    
    STORY BEATS (8–12 TOTAL)
    Each beat must advance tension through action or dialog. Escalate the conflict based on the story spark.
    
    ⭐ ENRICHED MACRO ACTIONS
    Each action beat should be a clear, visible movement WITH descriptive context.
    Structure: [verb] + [object] + [quality/manner]
    Examples:
    - "grasps cylinder" → "grasps heavy metal cylinder with both hands"
    - "steps forward" → "steps forward with determined stride"
    Rules:
    - Use the full 12-word allowance
    - Include physical qualities: weight, texture, temperature
    - Include manner: speed, force, direction
    - Keep actions macro-level (visible body movements, not subtle gestures)
    
    RESOLVE THE GOAL
    Stop immediately when the primary character goal is resolved or definitively failed.
    
    GROUND IN LOCATION & SHOW, DON'T TELL
    Keep the scene physically anchored. Express internal states strictly through observable physical behavior.

OUTPUT FORMAT:
Write in standard literary prose. Begin with the cold open (static snapshot only), insert the ******* COLD OPEN END **** marker, then write exactly 8-12 story beats. Dialog must appear every 2-3 beats.
'''



# ============================================================================
# TOPICAL SCENE PROMPTS (new pipeline)
# ============================================================================

TOPICAL_SEED_GENERATOR = f'''
🎓 TOPICAL SCENE SEED GENERATOR (EDUCATIONAL/DEBATE)
ROLE — Generate structured seeds for educational or debate scenes

⭐ SCENARIO TYPE SELECTION
If user specifies scenario type, use it. Otherwise select from:
    1. DEBATE: Two characters with opposing viewpoints
    2. TEACHING: Expert teaching a student/apprentice
    3. DISCUSSION: Collaborative exploration of a topic
    4. MENTORSHIP: Experienced guide helping novice through challenge

Selection: Use (current hour % 4) + 1 if not specified.

⭐ TOPIC SELECTION
If user specifies topic, use it. Otherwise generate from these categories:
- Philosophy/Ethics
- Science/Technology
- History/Politics
- Art/Culture
- Personal Development
- Social Issues
- Creative Process
- Professional Skills

Selection: Use (current minute % 8) + 1 if not specified.

{CHARACTERS}

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

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def run_prompt(prompt, system, pth):
    if not Path(pth).exists():
        result = llm_analyze_media(
            media="",
            prompt=prompt,
            system=system,
            max_tokens=8192)['analysis']
        with open(pth, 'w') as out_f:
            out_f.write(result)
        print(f'Wrote {pth}')
        return result
    else:
        print(f'{pth} Exists')
        return Path(pth).read_text()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-S', '--story', type=str, default='',
                        help='Path to existing story/seed file to expand')
    parser.add_argument('-O', '--output', type=str, default='story.txt',
                        help='Output file path')
    parser.add_argument('-T', '--topic', type=str, default=None,
                        help='Topic/focus (for topical: the subject; for action: the test focus)')
    parser.add_argument('-G', '--genre', type=str, default=None,
                        help='Genre (action mode only)')
    parser.add_argument('--topical', action='store_true',
                        help='Generate topical/educational scene (debate/teaching)')
    parser.add_argument('--scenario', type=str, default=None,
                        help='Scenario type for topical: DEBATE/TEACHING/DISCUSSION/MENTORSHIP')
    parser.add_argument('--seed-only', action='store_true',
                        help='Only generate seed, do not expand')
    args = parser.parse_args()

    if args.topical:
        # ====================================================================
        # TOPICAL PIPELINE: seed generation → scene expansion
        # ====================================================================
        output_path = Path(args.output)
        if args.story:
            seed = Path(args.story).read_text()
        else:
            seed_path = output_path.with_name(output_path.stem + '_seed.txt')

            # Build seed prompt
            inputs = 'Generate a topical seed\n'
            if args.topic:
                inputs += f'Topic: {args.topic}\n'
            else:
                inputs += f'Topic: {random.choice(TOPICAL_TOPICS)}\n'
            if args.scenario:
                if args.scenario.upper() in TOPICAL_SCENARIOS:
                    inputs += f'Scenario Type: {args.scenario.upper()}\n'
                else:
                    print(f'Warning: Unknown scenario "{args.scenario}", using random')
                    inputs += f'Scenario Type: {random.choice(TOPICAL_SCENARIOS)}\n'

            seed = run_prompt(inputs, TOPICAL_SEED_GENERATOR, str(seed_path))

        if not args.seed_only:
            # Expand seed into full topical scene
            print(run_prompt(seed, TOPICAL_EXPANDER, str(output_path)))
        else:
            print(f'Seed written to {seed_path}')

    elif args.story:
        # ====================================================================
        # DIRECT EXPANSION: expand existing story/seed file
        # ====================================================================
        story = Path(args.story).read_text()
        print(run_prompt(story, ALTERNATIVE, args.output))

    elif args.topic:
        # ====================================================================
        # ACTION SEED GENERATION: generate action seed then expand
        # ====================================================================
        output_path = Path(args.output)
        seed_path = output_path.with_name(output_path.stem + '_seed.txt')

        if args.topic.upper() in TOPICS:
            topic = args.topic.upper()
        else:
            topic = random.choice(TOPICS)

        genre = args.genre if args.genre else random.choice(GENRES)
        inputs = f'Generate a test seed\nGenre: {genre}\n Focus: {topic}'

        seed = run_prompt(inputs, SEED_GENERATOR, str(seed_path))

        if not args.seed_only:
            print(run_prompt(seed, ALTERNATIVE, str(output_path)))
        else:
            print(f'Seed written to {seed_path}')

    else:
        # ====================================================================
        # DEFAULT: use hardcoded SEED
        # ====================================================================
        print(run_prompt(SEED, ALTERNATIVE, args.output))