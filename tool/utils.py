import csv
import os
import time
from dotenv import load_dotenv

RESULTS_PATH = "Results/"

ROLE_FILE ="role.txt"
CONTEXT_FILE ="context.txt"
CLASSIFICATION_FILE ="classificationTask.txt"
FORMAT_FILE ="format.txt"
CONSTRAINTS_FILE ="constraints.txt"
FEWSHOT_FILE ="fewShot.txt"
ZEROSHOTCOT_FILE ="0ShotCoT.txt"
FEWSHOTCOT_FILE ="few-shotCoT.txt"

CLASSIFICATION_CAT_FILE ="classificationTaskCat.txt"
FORMAT_CAT_FILE ="formatCat.txt"


# API Key para OpenAI.
load_dotenv()  # load from .env file
OAI_2 = os.getenv("OAI_2")
GEMINI = os.getenv("GEMINI")
CLAUDE = os.getenv("CLAUDE")

HELP_ROLE = """You are an expert research assistant specializing in qualitative analysis of behavioral economics experiments, with extensive experience interpreting and coding free-form participant responses."""

HELP_CONTEXTO = """This experiment examines coordination between a manager and two agents under conditions of conflicting preferences and asymmetric information. Participants (567 undergraduate students in 189 groups of three) were assigned fixed roles (one manager M, two agents S1 and S2) and played 18 rounds where the game state varied randomly.

GAME STRUCTURE:
- All parties benefit when agents coordinate on a common action
- Agents have opposing preferences over which coordinated outcome to choose
- Agents observe the game state; the manager does not
- The efficient outcome (maximizing total surplus) varies by state
- Manager's objective is to maximize combined agent payoffs

TREATMENT CONDITIONS:
Groups were assigned to different conditions varying:

Communication type:
- No communication (baseline)
- Structured communication (predefined message options)
- Free-form chat (unrestricted text messaging)

Control mechanism:
- Delegation (agents choose actions)
- Delegation with managerial advice
- Managerial control (manager selects actions)

After each round, participants received feedback on the game state and actions. In some treatments, managers could observe discrepancies between agents' reported information and the actual state."""

HELP_CLASIFICACION = """Language note: The input messages are primarily in Spanish. Interpret the content in Spanish from Spain and output only the required labels in the specified format.

The task you have is to classify each group of messages sent in one or more of the following categories. The group of messages represent what the participants communicated sequentially during each period. Additionally, it states who sent the message: 0 if it is Central Manager, 1 if it is agent 1, and 2 if it is agent 2. 

These are the categories stated:

1. Did they make a suggestion about what row/column should be chosen? (any_suggestion) If it was the case:
a. Did they suggest a safe outcome? (suggest_safe)
b. Did they suggest an efficient outcome? (suggest_efficient)

2. Did they agree to the proposal about what row/column should be chosen? (agree_proposal)
3. Did they discuss about what row/column should be chosen? If it was the case:
a. Did they discuss need to coordinate? (discuss_coordinate) IMPORTANT BOUNDARY: This requires more than making a suggestion that involves coordination. The message needs to indicate that the two players should be choosing the same thing (e.g. "We'll do better if we make the same choices" is coded. "Let's choose row 4 and column 4" is not coded.)   
b. Did they discuss fairness? (discuss_fairness) This category includes any message that discusses the distribution of pay over the three players. Use 0.5 if fairness is raised only briefly or incidentally.     
c. Did they discuss efficiency? (discuss_efficient) This includes discussion of maximizing total pay as well as explaining how and why rotation between players works. Use 0.5 if efficiency is mentioned in passing.
4. Did they send questions about rules of the experiment? (discuss_rules) Use 0.5 if the question is ambiguous (could be asking about rules or about strategy).
5. Did they give any explanation (explanation)? This included explanations about the rules of the experiment or game, as well as explanations of a suggested way of playing the game. Use 0.5 if the explanation is partial or unclear.
6. Did they send questions about how to play? (discuss_howtoplay) This was for conceptual questions rather than the frequent generic request that somebody suggest a row and column.
7. Did the Manager (coded as 0) ask what game is being Played? (ask_game) Use 0.5 if the question is implicit or ambiguous.
8. Did Agents Report What Game is Being Played? (receive_report) If that is the case:
a. Did they truthfully Reveal Game? (truthful)
b. Did they lie About Game? (falsehood)
c. Conflict (contradict): The agents contradict each other — one says one game, the other says a different game. (Note: previously mislabeled 'contradict' was the fact-checking category; see constraint file for final mapping.)
9. None of the agents report. This is true if in the last question "Did Agents Report What Game is Being Played?", the answer was false. (neither_report)"""

HELP_FORMAT = """Please provide your answer strictly as a Python dictionary with no additional text or characters.

Use these exact keys with no extra spaces:
- 'any_suggestion'
- 'suggest_safe'
- 'suggest_efficient'
- 'agree_proposal'
- 'discuss_coordinate'
- 'discuss_fairness'
- 'discuss_efficient'
- 'discuss_rules'
- 'explanation'
- 'discuss_howtoplay'
- 'ask_game'
- 'receive_report'
- 'truthful'
- 'falsehood'
- 'contradict'
- 'neither_report'

All values must be:
- 1 if the category is clearly observed
- 0 if the category is not observed

Be sure that each tag is in simple quotes. 
Example format:
{
    'any_suggestion': 1,
    'suggest_safe': 0,
    'suggest_efficient': 1,
    'agree_proposal': 0,
    'discuss_coordinate': 1,
    'discuss_fairness': 1,
    'discuss_efficient': 0,
    'discuss_rules': 1,
    'explanation': 1,
    'discuss_howtoplay': 1,
    'ask_game': 0,
    'receive_report': 0,
    'truthful': 0,
    'falsehood': 0,
    'contradict': 0,
    'neither_report': 1
}
"""

HELP_CONSTRAINTS = """LOGICAL CONSTRAINTS:

1. Suggestion hierarchy:
   - If 'any_suggestion': 0, then both 'suggest_safe': 0 and 'suggest_efficient': 0
   - If 'any_suggestion': 1, then 'suggest_safe' and 'suggest_efficient' can be 0 or 1 independently

2. Reporting logic:
   - 'receive_report' and 'neither_report' are mutually exclusive:
     • If 'receive_report': 0, then 'neither_report': 1
     • If 'receive_report': 1, then 'neither_report': 0

3. Report content (only applies when 'receive_report': 1):
   - Exactly one of the following must equal 1, the others must be 0:
     • 'truthful'
     • 'falsehood'
     • 'contradict'
   - If 'receive_report': 0, then all three must be 0
"""

HELP_COT = """Apart from the classification task, please state a one-sentence justification or rationale behind your classification  (máx. 25–40 WORDS). In such sense, your answer should be:

{
'any_suggestion': 1, 
'suggest_safe ': 0,
'suggest_efficient': 1,
'agree_proposal' : 1,
'discuss_coordinte': 0,
'discuss_fairness': 0.5,
'discuss_efficient': 0,
'discuss_rules': 0,
'explanation': 0,
'discuss_howtoplay': 0,
'ask_game': 0,
'receive_report': 0,
'truthful': 0,
'falsehood': 0,
'contradict': 0,
'neither_report': 1,
'reason': <here you write the rationale behind your decisions>'
}
"""

HELP_FS_COT = """Apart from the classification task, please state a one-sentence justification or rationale behind your classification  (máx. 25–40 WORDS). In such sense, your answer should be:

{
'any_suggestion': 1, 
'suggest_safe ': 0,
'suggest_efficient': 1,
'agree_proposal' : 1,
'discuss_coordinte': 0,
'discuss_fairness': 1,
'discuss_efficient': 0,
'discuss_rules': 0,
'explanation': 0,
'discuss_howtoplay': 0,
'ask_game': 0,
'receive_report': 0,
'truthful': 0,
'falsehood': 0,
'contradict': 0,
'neither_report': 1,
'reason': <here you write the reasons behind your decisions>'
}

The following are some examples based on real responses of participants:

The group of messages is:

1; ui mira, el 5 / 1; jajajajajjajajajajjaja / 2; jajaja / 2; c3 f3 entonces / 1; c4 f4 era / 2; jajaja si / 2; c4f4 / 1; valee / 1; no me la trames ahora ¬¬ / 2; no no jajaja / 2; soy de fiar / 1; me lo creeré jajaja / 2; ya veras / 1; :) / 2; c1f1 / 1; c2-f2 esta bien / 1; ostras tu jajaja menuda diferencia / 2; yo prefiero c1f1, que llevas ganando tu dos / 1; bueno vale / 2; y el proximo iguales / 2; si nos toca el juego tres perfecto / 1; ya ves / 1; jajajajaja haz fuerza / 2; jajajaja valee / 2; que hacemos? yo ya me he perdido / 1; jajajaj aqui en realidad nos renta ir iguales porque son 40, tu aqui tu maximop ara llevarte son 46 / 2; f3c3? / 2; y ganamos los 2 lo mismo / 1; pero si se trata de ganar lo max / 2; no sabemos en la siguiente ronda que nos va a salir+ / 1; vamos compensando cada ronda / 1; ya pero los numeros son los mismos en todas las tablas / 1; solo cambia el orden / 1; buscamos las combinaciones / 2; okey , en la siguiente ronda , gano yo y pierdes más tu entonces / 1; f2 c2? / 1; pierdo 3 ganas 11 / 2; c1 y f1 / 1; ahi pierdo yo mas de 10 / 2; c2 / 2; f2 / 2; c2 / 2; f2 / 1; va / 1; c1 f 1 / 0; c3 f3 / 1; sii eso A / 2; ENCIMA TAMBN ES LA Q UE MAS LE DAN AL a / 0; esa definitiva / 1; al A le dan mucho / 1; que suerte / 2; flipas / 2; jajaja / 1; que me compense ahora / 1; no? / 2; creo q no hay / 2; si / 1; que eficacia jajaja / 0; si jajaja / 1; o mejor 22

The classification should be:
{
'any_suggestion': 1,
'suggest_safe': 1,
'suggest_efficient': 0,
'agree_proposal': 1,
'discuss_fairness': 0,
'discuss_efficient': 0,
'discuss_rules': 0,
'explanation': 0,
'discuss_howtoplay': 0,
'ask_game': 0,
'receive_report': 0,
'truthful': 0,
'falsehood': 0,
'contradict': 0,
'neither_report': 1
'reason': 'They repeatedly propose and agree on specific row/column combinations, explicitly discuss coordinating and equalizing payoffs across rounds, compare maximum gains, and explain how tables work, but no one reports which game is being played.'
}

The group of messages is:

1; di tu primero / 2; c1 / 1; f1 / 2; vale bien / 1; mejor la f5 / 2; ya estaba decidido jajaj / 2; si / 1; f3 / 2; ganas mas con el f1 / 1; c3-f2 esta bien / 2; uuuf / 2; que miedo / 2; jajajajja / 1; en el bloque c5 / 2; ajajaja / 1; por? / 2; un poco tarde ya / 2; porque es la suma mas alta / 1; entonces que / 2; ahora ya no, no? / 1; me da uq eno / 1; iguales? / 2; okey , estamos deacuerdo / 1; menosmal :P / 2; c1 / 1; f1 / 1; mira tengo una teoria / 2; iluminame / 1; en el juego 1 f1 c1 ganas tu / 1; juego dos f2 c2 ganas mas tu / 2; jajaja / 2; BUENA / 0; bien jugada / 1; vamos / 0; sigamos / 1; pues esa / 0; por ahira c3 f3 / 1; si / 2; leugo hacemos una / 2; esque hacer eso compensa si luego lo hacemos al reves / 2; si / 1; vale pues ya estmos iguales / 2; tenemos q ir haciendo eso q se va turnando / 1; columna4 fila 4 / 0; valeç / 0; siempre lo mismo? / 1; no / 1; mira las variantes / 1; hay que ver que nos sume el maximo de puntos a cada uno / 1; para poder tener mas ganancias / 0; vale ya lo entiendo / 1; en las casillas que haya negativo no le des / 1; porque nos afecta y perdemos dinero / 1; el juego es el 5 / 2; fila 4 / 1; columna 4 fila 4 / 2; si / 0; si / 1; ganamos todos

The classification should be:
{
'any_suggestion': 1,
'suggest_safe': 0,
'suggest_efficient': 1,
'agree_proposal': 1,
'discuss_fairness': 1,
'discuss_efficient': 0,
'discuss_rules': 0,
'explanation': 0,
'discuss_howtoplay': 0,
'ask_game': 0,
'receive_report': 0,
'truthful': 0,
'falsehood': 0,
'contradict': 0,
'neither_report': 1,
'reason': 'Participants repeatedly proposed specific row-column combinations and explicitly agreed (e.g., “columna4 fila4”, “si”), showing coordinated suggestions; they discussed alternating gains and equalizing outcomes, indicating partial fairness concerns without explicit rule clarification or state reporting.'
}

The group of messages is:

0; ya eligo siempre igual / 0; os parece bien no? / 0; juego 1? / 1; si / 2; si,claro / 0; c3 f3 / 0; os parece? / 2; sii / 0; pues c3 f3 / 2; 3 / 0; juego / 2; pongo la que quiera / 1; elegimos columna 5 y fila 5 / 1; ? / 0; nos ponemos de acuerdo para no perder ninguno / 2; noo / 2; que es negativa la 5,5 / 0; juego? / 2; ahora es juego 3 / 1; ha sido un placer / 1; ajaj / 1; suerte / 1; || / 2; como el comunismo / 0; igual / 1; !! / 2; un placer / 0; :) a por todas / 2; como chocolate valor / 2; ummm / 1; ahora va otra ronda? / 0; sii / 2; con otras pers / 0; de 9 / 1; cuantas rondas hay? / 2; 9 / 1; 18 / 0; que va para eso se va de fiesta / 2; si F / 0; sii / 2; mareee / 1; esta tarde una cerveza y se te pasa / 2; o una buena merienda / 2; un bocata de nocilla / 2; ganamos todos mas / 2; en esta ronda / 1; cierto / 2; ponemos eso? / 0; vale, pues así / 0; f2 c2 / 2; oki / 1; ok / 1; 3 / 1; este es un lujo / 2; cual ponemos? / 0; valee

The classification should be:

{
'any_suggestion': 1,
'suggest_safe': 1,
'suggest_efficient': 0,
'agree_proposal': 0,
'discuss_fairness': 0,
'discuss_efficient': 0,
'discuss_rules': 0,
'explanation': 0,
'discuss_howtoplay': 0,
'ask_game': 0,
'receive_report': 1,
'truthful': 1,
'falsehood': 0,
'contradict': 0,
'neither_report': 0,
'reason': 'They propose specific row/column choices, agree on them, and explicitly mention coordinating to avoid losses and earn more; the manager asks which game it is and one agent reports “game 3” without contradiction, plus questions about rounds indicate rule discussion.'
}"""


HELP_FS = """The following are some examples based on real responses of participants:

The group of messages is:

1; ui mira, el 5 / 1; jajajajajjajajajajjaja / 2; jajaja / 2; c3 f3 entonces / 1; c4 f4 era / 2; jajaja si / 2; c4f4 / 1; valee / 1; no me la trames ahora ¬¬ / 2; no no jajaja / 2; soy de fiar / 1; me lo creeré jajaja / 2; ya veras / 1; :) / 2; c1f1 / 1; c2-f2 esta bien / 1; ostras tu jajaja menuda diferencia / 2; yo prefiero c1f1, que llevas ganando tu dos / 1; bueno vale / 2; y el proximo iguales / 2; si nos toca el juego tres perfecto / 1; ya ves / 1; jajajajaja haz fuerza / 2; jajajaja valee / 2; que hacemos? yo ya me he perdido / 1; jajajaj aqui en realidad nos renta ir iguales porque son 40, tu aqui tu maximop ara llevarte son 46 / 2; f3c3? / 2; y ganamos los 2 lo mismo / 1; pero si se trata de ganar lo max / 2; no sabemos en la siguiente ronda que nos va a salir+ / 1; vamos compensando cada ronda / 1; ya pero los numeros son los mismos en todas las tablas / 1; solo cambia el orden / 1; buscamos las combinaciones / 2; okey , en la siguiente ronda , gano yo y pierdes más tu entonces / 1; f2 c2? / 1; pierdo 3 ganas 11 / 2; c1 y f1 / 1; ahi pierdo yo mas de 10 / 2; c2 / 2; f2 / 2; c2 / 2; f2 / 1; va / 1; c1 f 1 / 0; c3 f3 / 1; sii eso A / 2; ENCIMA TAMBN ES LA Q UE MAS LE DAN AL a / 0; esa definitiva / 1; al A le dan mucho / 1; que suerte / 2; flipas / 2; jajaja / 1; que me compense ahora / 1; no? / 2; creo q no hay / 2; si / 1; que eficacia jajaja / 0; si jajaja / 1; o mejor 22

The classification should be:
{
'any_suggestion': 1,
'suggest_safe': 1,
'suggest_efficient': 0,
'agree_proposal': 1,
'discuss_fairness': 0,
'discuss_efficient': 0,
'discuss_rules': 0,
'explanation': 0,
'discuss_howtoplay': 0,
'ask_game': 0,
'receive_report': 0,
'truthful': 0,
'falsehood': 0,
'contradict': 0,
'neither_report': 1
}

The group of messages is:

1; di tu primero / 2; c1 / 1; f1 / 2; vale bien / 1; mejor la f5 / 2; ya estaba decidido jajaj / 2; si / 1; f3 / 2; ganas mas con el f1 / 1; c3-f2 esta bien / 2; uuuf / 2; que miedo / 2; jajajajja / 1; en el bloque c5 / 2; ajajaja / 1; por? / 2; un poco tarde ya / 2; porque es la suma mas alta / 1; entonces que / 2; ahora ya no, no? / 1; me da uq eno / 1; iguales? / 2; okey , estamos deacuerdo / 1; menosmal :P / 2; c1 / 1; f1 / 1; mira tengo una teoria / 2; iluminame / 1; en el juego 1 f1 c1 ganas tu / 1; juego dos f2 c2 ganas mas tu / 2; jajaja / 2; BUENA / 0; bien jugada / 1; vamos / 0; sigamos / 1; pues esa / 0; por ahira c3 f3 / 1; si / 2; leugo hacemos una / 2; esque hacer eso compensa si luego lo hacemos al reves / 2; si / 1; vale pues ya estmos iguales / 2; tenemos q ir haciendo eso q se va turnando / 1; columna4 fila 4 / 0; valeç / 0; siempre lo mismo? / 1; no / 1; mira las variantes / 1; hay que ver que nos sume el maximo de puntos a cada uno / 1; para poder tener mas ganancias / 0; vale ya lo entiendo / 1; en las casillas que haya negativo no le des / 1; porque nos afecta y perdemos dinero / 1; el juego es el 5 / 2; fila 4 / 1; columna 4 fila 4 / 2; si / 0; si / 1; ganamos todos


The classification should be:
{
'any_suggestion': 1,
'suggest_safe': 0,
'suggest_efficient': 1,
'agree_proposal': 1,
'discuss_fairness': 1,
'discuss_efficient': 0,
'discuss_rules': 0,
'explanation': 0,
'discuss_howtoplay': 0,
'ask_game': 0,
'receive_report': 0,
'truthful': 0,
'falsehood': 0,
'contradict': 0,
'neither_report': 1,
}

The group of messages is:

0; ya eligo siempre igual / 0; os parece bien no? / 0; juego 1? / 1; si / 2; si,claro / 0; c3 f3 / 0; os parece? / 2; sii / 0; pues c3 f3 / 2; 3 / 0; juego / 2; pongo la que quiera / 1; elegimos columna 5 y fila 5 / 1; ? / 0; nos ponemos de acuerdo para no perder ninguno / 2; noo / 2; que es negativa la 5,5 / 0; juego? / 2; ahora es juego 3 / 1; ha sido un placer / 1; ajaj / 1; suerte / 1; || / 2; como el comunismo / 0; igual / 1; !! / 2; un placer / 0; :) a por todas / 2; como chocolate valor / 2; ummm / 1; ahora va otra ronda? / 0; sii / 2; con otras pers / 0; de 9 / 1; cuantas rondas hay? / 2; 9 / 1; 18 / 0; que va para eso se va de fiesta / 2; si F / 0; sii / 2; mareee / 1; esta tarde una cerveza y se te pasa / 2; o una buena merienda / 2; un bocata de nocilla / 2; ganamos todos mas / 2; en esta ronda / 1; cierto / 2; ponemos eso? / 0; vale, pues así / 0; f2 c2 / 2; oki / 1; ok / 1; 3 / 1; este es un lujo / 2; cual ponemos? / 0; valee

The classification should be:

{
'any_suggestion': 1,
'suggest_safe': 1,
'suggest_efficient': 0,
'agree_proposal': 0,
'discuss_fairness': 0,
'discuss_efficient': 0,
'discuss_rules': 0,
'explanation': 0,
'discuss_howtoplay': 0,
'ask_game': 0,
'receive_report': 1,
'truthful': 1,
'falsehood': 0,
'contradict': 0,
'neither_report': 0
}"""


HELP_CLASIFICACION_CAT = """
Your task is to analyze the full set of chat messages provided and identify three major thematic categories that consistently appear throughout the messages. These categories should capture the dominant topics or communication strategies participants use across the experiment. A message may belong to more than one category.
"""

HELP_FORMAT_CAT = """
You must return the three categories you identify strictly as a Python dictionary, with no additional text or characters.
Each key should be a descriptive tag, meaning that each must accurately depict the category you chose.  Also,  each value should be a definition of the category.
For example, if your categories are A, B, and C, and your tags are 'A', 'B', and 'C', the response should be:

 {
    'A': 'the category is A: <definition of the category> ',
    'B': 'the category is B: <definition of the category> ',
    'C': 'the category is C: <definition of the category> '
}

You must:
1.	Analyze the complete content.
2.	Identify the three most prevalent thematic categories.
3.	Return them only in the required Python dictionary format.

After this first step (creating the categories), further instructions will follow for applying them to each individual message.

"""


# Model lists are derived from one shared capability registry.
from model_config import models_for
gpt_models = models_for("chatgpt")
gemini_models = models_for("gemini")
claude_models = models_for("claude")
