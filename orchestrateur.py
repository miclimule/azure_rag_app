import os
import sys
import json
import urllib.request
from dotenv import load_dotenv
from openai import AzureOpenAI
from agent_image.image_agent_prediction import call_image_agent

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

model_name = os.getenv("AI_MODEL_DEPLOYMENT_NAME")
deployment = os.getenv("AI_MODEL_DEPLOYMENT")

LLM = AzureOpenAI(
    api_version=os.getenv("AZURE_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_AI_ENDPOINT"),
    api_key=os.getenv("AZURE_AI_KEY"),
)

def detect_intents(message: str) -> list[dict]:
    available_agents = [
        {"agent": "document_agent", "description": "Pour obtenir des informations documentaires, notamment sur les plantes, objets ou données textuelles."},
        {"agent": "weather_agent", "description": "Pour récupérer les prévisions météorologiques ou les conditions climatiques."},
        {"agent": "image_agent", "description": "Analyse l'image de la plante pour connaître son état, si elle est malade ou pas."},
        {"agent": "general", "description": "Pour les cas généraux où aucun agent spécifique ne convient."}
    ]

    prompt = (
        f"Voici la liste des agents disponibles : {json.dumps(available_agents, ensure_ascii=False)}\n"
        "Tu es un routeur intelligent. Selon le message utilisateur, retourne une liste JSON ordonnée avec les agents à appeler.\n"
        "Chaque élément de la liste doit être un objet avec :\n"
        "- 'agent': le nom de l'agent\n"
        "- 'reason': une phrase expliquant pourquoi cet agent est utile\n"
        "Exemple : [{\"agent\": \"document_agent\", \"reason\": \"pour connaître les besoins d'arrosage\"}, {\"agent\": \"weather_agent\", \"reason\": \"pour savoir s'il va pleuvoir\"}, {\"agent\": \"image_agent\", \"reason\": \"pour connaître la maladie de la plante\"}]\n"
        f"Message utilisateur : {message}"
    )

    response = LLM.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "Tu es un planificateur intelligent qui détermine les agents à appeler, dans l'ordre, avec une raison."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0
    )

    try:
        intents = json.loads(response.choices[0].message.content.strip())
        if isinstance(intents, list):
            return intents
    except:
        pass
    return [{"agent": "general", "reason": "aucun agent spécifique détecté"}]

def call_general_agent(message: str) -> str:
    response = LLM.chat.completions.create(
        messages=[
            {"role": "system", "content": "Tu es un assistant intelligent expert dans la gestion de plante"},
            {"role": "user", "content": message}
        ],
        max_tokens=1024,
        temperature=1.0,
        top_p=1.0,
        model=deployment,
    )
    return response.choices[0].message.content

def download_image(url: str, save_path: str = "temp_image.jpg") -> str:
    try:
        urllib.request.urlretrieve(url, save_path)
        return save_path
    except Exception as e:
        print(f"Erreur lors du téléchargement de l'image : {e}")
        return None

def extract_image_path(message: str) -> str:
    if message.startswith("http://") or message.startswith("https://"):
        return download_image(message)
    elif os.path.isfile(message):
        return message
    else:
        return None

def orchestrate(user_message: str):
    agents_plan = detect_intents(user_message)
    context = {}

    for step in agents_plan:
        agent = step.get("agent")
        if agent == "document_agent":
            context["document"] = "Il faut arroser les plantes"
        elif agent == "weather_agent":
            context["weather"] = "Il va pleuvoir dans 6 heures"
        elif agent == "image_agent":
            image_path = extract_image_path(user_message)
            if image_path:
                context["image"] = call_image_agent(image_path)
            else:
                context["image"] = "Aucune image valide trouvée dans le message utilisateur."
        elif agent == "general":
            context["general"] = call_general_agent(user_message)

    context_prompt = "Voici les informations obtenues des différents agents :\n"
    for key, value in context.items():
        context_prompt += f"[{key.upper()}]: {value}\n"
    context_prompt += "\nEn te basant uniquement sur ces informations, réponds à la question de l'utilisateur :\n"
    context_prompt += user_message

    response = LLM.chat.completions.create(
        model=deployment,
        stream=True,
        messages=[
            {"role": "system", "content": "Tu es un assistant intelligent expert dans la gestion de plante qui synthétise des réponses à partir des résultats d'autres agents."},
            {"role": "user", "content": context_prompt}
        ],
        max_tokens=1024,
        temperature=0.7
    )

    for update in response:
        if update.choices:
            print((update.choices[0].delta.content or "").encode('utf-8', errors='ignore').decode('utf-8'), end="")

if __name__ == "__main__":
    message = input("Message: ")
    orchestrate(message)



























import os
import re
import json
import logging
import tempfile
import urllib.request
from typing import List, Dict, Optional
from dotenv import load_dotenv

# LangChain imports
from langchain.chat_models import AzureChatOpenAI
from langchain import LLMChain, PromptTemplate

# Ton agent d'image existant (conserve l'API que tu as déjà)
from agent_image.image_agent_prediction import call_image_agent

load_dotenv()
logging.basicConfig(level=logging.INFO)

# Azure / LangChain LLM wrapper
LLM = AzureChatOpenAI(
    deployment_name=os.getenv("AI_MODEL_DEPLOYMENT_NAME") or os.getenv("AI_MODEL_DEPLOYMENT"),
    openai_api_key=os.getenv("AZURE_AI_KEY"),
    openai_api_base=os.getenv("AZURE_AI_ENDPOINT"),
    openai_api_version=os.getenv("AZURE_API_VERSION"),
    temperature=0,
)

# Agents disponibles (déclaratif, lisible par le modèle)
AVAILABLE_AGENTS = [
    {"agent": "document_agent", "description": "infos documentaires sur plantes, guides, fiches techniques"},
    {"agent": "weather_agent", "description": "prévisions météo et conditions locales"},
    {"agent": "image_agent", "description": "analyse d'image de plante pour détecter maladie / état"},
    {"agent": "general", "description": "réponse générique ou synthèse"},
]

# Prompt pour détecter les intents et ordonner les agents
DETECT_PROMPT = PromptTemplate(
    input_variables=["message", "agents_json"],
    template=(
        "Tu es un routeur d'agents. Tu reçois un message utilisateur et renvoies une liste JSON ordonnée "
        "des agents à appeler. Chaque élément doit être un objet avec 'agent' et 'reason'.\n\n"
        "Agents disponibles: {agents_json}\n\n"
        "Message utilisateur: {message}\n\n"
        "Réponds uniquement par JSON. Exemple: [{\"agent\": \"document_agent\", \"reason\": \"pour connaître les besoins d'arrosage\"}]"
    ),
)

DETECT_CHAIN = LLMChain(llm=LLM, prompt=DETECT_PROMPT)

GENERAL_PROMPT = PromptTemplate(
    input_variables=["message"],
    template=(
        "Tu es un assistant expert en gestion de plantes. Réponds précisément et de façon concise "
        "au message utilisateur suivant en te limitant aux informations demandées:\n\n{message}"
    ),
)
GENERAL_CHAIN = LLMChain(llm=LLM, prompt=GENERAL_PROMPT)

FINAL_PROMPT = PromptTemplate(
    input_variables=["context", "user_message"],
    template=(
        "Tu es un assistant qui synthétise les résultats d'autres agents. Voici les informations obtenues:\n\n{context}\n\n"
        "En te basant uniquement sur ces informations, fournis une réponse claire, structurée et concise à la demande de l'utilisateur:\n\n{user_message}"
    ),
)
FINAL_CHAIN = LLMChain(llm=LLM, prompt=FINAL_PROMPT)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}

URL_REGEX = re.compile(r"https?://[^\s'\">]+", re.IGNORECASE)
WINDOWS_PATH_REGEX = re.compile(r"[A-Za-z]:[/\\][^\s'\">]+")


def extract_image_sources(text: str) -> List[Dict[str, str]]:
    """Extrait toutes les sources d'images du texte.
    Retourne une liste de dicts {"type": "url"|"file", "value": <str>}.
    Les 'file' peuvent être des fichiers ou des dossiers."""
    if not text:
        return []

    sources: List[Dict[str, str]] = []

    # 1) URLs
    for m in URL_REGEX.findall(text):
        # nettoyer ponctuation finale
        clean = re.sub(r"[)\]\.,;!?]+$", "", m)
        sources.append({"type": "url", "value": clean})

    # 2) Windows-like paths explicitement écrits
    for m in WINDOWS_PATH_REGEX.findall(text):
        clean = m.strip("'\"")
        sources.append({"type": "file", "value": os.path.normpath(clean)})

    # 3) Tokens that look like local unix paths or relative paths
    tokens = re.split(r"[\s,;]+", text)
    for t in tokens:
        t = t.strip("'\"`")
        if not t:
            continue
        # si c'est déjà détecté, skip
        if any(t == s["value"] for s in sources):
            continue
        # heuristique: contient slash and no protocol
        if (t.startswith("/") or t.startswith("./") or t.startswith("..") or \
            re.search(r"[\\/][^\\/]+", t)):
            # normaliser
            norm = os.path.normpath(t)
            sources.append({"type": "file", "value": norm})

    # 4) si aucun source détecté, retourne vide
    return sources


def collect_files_from_path(path: str) -> List[str]:
    """Si path est fichier valide, retourne [file]. Si dossier, explore récursivement et retourne les images.
    Si path n'existe pas, retourne empty list."""
    found: List[str] = []
    try:
        if os.path.isfile(path):
            ext = os.path.splitext(path)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                found.append(path)
            return found
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for f in files:
                    if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                        found.append(os.path.join(root, f))
            return found
    except Exception as e:
        logging.exception("Erreur en parcourant le chemin local %s: %s", path, e)
    return found


def download_image(url: str) -> Optional[str]:
    """Télécharge l'image vers un fichier temporaire et retourne le chemin."""
    try:
        suffix = os.path.splitext(url.split("?")[0])[-1]
        if not suffix or len(suffix) > 5:
            suffix = ".jpg"
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        urllib.request.urlretrieve(url, tf.name)
        logging.info("Image téléchargée: %s", tf.name)
        return tf.name
    except Exception as e:
        logging.exception("Erreur téléchargement image %s: %s", url, e)
        return None


def detect_intents(user_message: str) -> List[Dict[str, str]]:
    """Retourne la liste ordonnée d'agents à appeler, déduite par le LLM (JSON)."""
    prompt_inputs = {
        "message": user_message,
        "agents_json": json.dumps(AVAILABLE_AGENTS, ensure_ascii=False),
    }
    raw = DETECT_CHAIN.run(prompt_inputs)
    try:
        parsed = json.loads(raw.strip())
        if isinstance(parsed, list):
            return parsed
    except Exception:
        logging.warning("Impossible de parser la sortie du détecteur d'intent. Sortie brute: %s", raw)
    return [{"agent": "general", "reason": "aucun agent spécifique détecté"}]


def call_general_agent(message: str) -> str:
    return GENERAL_CHAIN.run({"message": message})


def orchestrate(user_message: str) -> Dict[str, object]:
    """Orchestre les appels aux agents et renvoie un dict structuré de résultats.

    - Prend en charge plusieurs images trouvées dans le message.
    - Supporte URLs, fichiers locaux et dossiers locaux (ex: C:/images ou /home/user/images).
    - Retourne la liste des résultats d'analyse image dans context['images']."""
    plan = detect_intents(user_message)
    context: Dict[str, object] = {}

    # Extraire les sources d'images mentionnées dans le message
    image_sources = extract_image_sources(user_message)
    logging.info("Sources d'images détectées: %s", image_sources)

    for step in plan:
        agent = step.get("agent")
        reason = step.get("reason", "")
        logging.info("Step agent: %s — %s", agent, reason)

        if agent == "document_agent":
            # TODO: remplacer par ton vrai appel au document_agent
            context["document"] = "[simulation] fiche documentaire: arroser 2x/semaine"

        elif agent == "weather_agent":
            # TODO: remplacer par ton vrai appel au weather_agent
            context["weather"] = "[simulation] pluie possible dans 6h"

        elif agent == "image_agent":
            images_results = []

            # Si l'utilisateur n'a fourni aucun source mais que le message contient un chemin relatif qui existe,
            # extract_image_sources l'aura déjà trouvé. Sinon on peut vérifier un dossier par défaut si besoin.
            if not image_sources:
                context["image"] = "Aucune source d'image détectée dans le message utilisateur."
                logging.info("Aucune source d'image trouvée pour image_agent")
            else:
                for src in image_sources:
                    src_type = src.get("type")
                    src_value = src.get("value")
                    logging.info("Traitement source: %s -> %s", src_type, src_value)

                    local_paths: List[str] = []

                    if src_type == "url":
                        dl = download_image(src_value)
                        if dl:
                            local_paths = [dl]
                        else:
                            images_results.append({"source": src_value, "error": "échec téléchargement"})
                            continue

                    elif src_type == "file":
                        # si chemin absolu ou relatif sur machine
                        # si path existe et est fichier ou dossier
                        if os.path.exists(src_value):
                            # collecte fichiers image si dossier
                            local_paths = collect_files_from_path(src_value)
                            if not local_paths and os.path.isfile(src_value):
                                # peut-être extension non standard, on l'ajoute
                                local_paths = [src_value]
                        else:
                            # tenter d'interpréter comme chemin windows avec backslashes
                            alt = src_value.replace('/', os.sep).replace('\\', os.sep)
                            if os.path.exists(alt):
                                local_paths = collect_files_from_path(alt)
                            else:
                                images_results.append({"source": src_value, "error": "chemin local introuvable"})
                                continue

                    # Appel à l'image agent pour chaque fichier local
                    for p in local_paths:
                        try:
                            logging.info("Appel image agent sur %s", p)
                            res = call_image_agent(p)
                            images_results.append({"source": p, "result": res})
                        except Exception as e:
                            logging.exception("Erreur appel image agent pour %s: %s", p, e)
                            images_results.append({"source": p, "error": str(e)})

                context["images"] = images_results

        elif agent == "general":
            context["general"] = call_general_agent(user_message)

        else:
            logging.warning("Agent inconnu demandé: %s", agent)
            context.setdefault("other", []).append({"agent": agent, "note": "agent inconnu"})

    # Préparer le contexte structuré pour la synthèse finale
    context_serialized = "\n".join([f"[{k.upper()}]: {v}" for k, v in context.items()])

    final_answer = FINAL_CHAIN.run({"context": context_serialized, "user_message": user_message})

    return {"plan": plan, "context": context, "final_answer": final_answer}


if __name__ == "__main__":
    msg = input("Message: ")
    result = orchestrate(msg)
    print("\n\n--- RÉSULTAT STRUCTURÉ ---\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))






























































































# import os
# import sys
# import json
# from dotenv import load_dotenv
# from openai import AzureOpenAI
# from agent_image.image_agent_prediction import call_image_agent

# sys.stdout.reconfigure(encoding='utf-8')
# load_dotenv()

# model_name = os.getenv("AI_MODEL_DEPLOYMENT_NAME")
# deployment = os.getenv("AI_MODEL_DEPLOYMENT")

# LLM = AzureOpenAI(
#     api_version=os.getenv("AZURE_API_VERSION"),
#     azure_endpoint=os.getenv("AZURE_AI_ENDPOINT"),
#     api_key=os.getenv("AZURE_AI_KEY"),
# )

# # Point d'amelioration :
#     # Les agents devrait pouvoir s'appeler entre eux
#     # Les prochain agent doivent avoir les reponses des agents precedents
#     # Il faudrait envoyer au agent une description de ce qu'il doivent faire

# def last_action(message: str):
#     return "laste action"

# def detect_intents(message: str) -> list[dict]:
#     available_agents = [
#         {"agent": "document_agent", "description": "Pour obtenir des informations documentaires, notamment sur les plantes, objets ou données textuelles."},
#         {"agent": "weather_agent", "description": "Pour récupérer les prévisions météorologiques ou les conditions climatiques."},
#         {"agent": "image_agent", "description": "Analyse l'image de la plante pour connaître son état, si elle est malade ou pas."},
#         {"agent": "general", "description": "Pour les cas généraux où aucun agent spécifique ne convient."}
#     ]

#     prompt = (
#         f"Voici la liste des agents disponibles : {json.dumps(available_agents, ensure_ascii=False)}\n"
#         "Tu es un routeur intelligent. Selon le message utilisateur, retourne une liste JSON ordonnée avec les agents à appeler.\n"
#         "Chaque élément de la liste doit être un objet avec :\n"
#         "- 'agent': le nom de l'agent\n"
#         "- 'reason': une phrase expliquant pourquoi cet agent est utile\n"
#         "Exemple : [{\"agent\": \"document_agent\", \"reason\": \"pour connaître les besoins d'arrosage\"}, {\"agent\": \"weather_agent\", \"reason\": \"pour savoir s'il va pleuvoir\"}, {\"agent\": \"image_agent\", \"reason\": \"pour connaître la maladie de la plante\"}]\n"
#         f"Message utilisateur : {message}"
#     )

#     response = LLM.chat.completions.create(
#         model=deployment,
#         messages=[
#             {"role": "system", "content": "Tu es un planificateur intelligent qui détermine les agents à appeler, dans l'ordre, avec une raison."},
#             {"role": "user", "content": prompt}
#         ],
#         max_tokens=300,
#         temperature=0
#     )

#     try:
#         intents = json.loads(response.choices[0].message.content.strip())
#         print("intents : ",intents)
#         if isinstance(intents, list):
#             return intents
#     except:
#         pass
#     return [{"agent": "general", "reason": "aucun agent spécifique détecté"}]

# def call_general_agent(message: str) -> str:
#     response = LLM.chat.completions.create(
#         messages=[
#             {"role": "system", "content": "Tu es un assistant intelligent expert dans la gestion de plante"},
#             {"role": "user", "content": message}
#         ],
#         max_tokens=1024,
#         temperature=1.0,
#         top_p=1.0,
#         model=deployment,
#     )

#     return response.choices[0].message.content

# def orchestrate(user_message: str):
#     agents_plan = detect_intents(user_message)
#     context = {}

#     for step in agents_plan:
#         agent = step.get("agent")
#         if agent == "document_agent":
#             # context["document"] = call_document_agent(user_message)
#             context["document"] = "Il faut arroser les plantes"
#         elif agent == "weather_agent":
#             # context["weather"] = call_weather_agent(user_message)
#             context["weather"] = "Il va pleuvoir dans 6 heures"
#         elif agent == "image_agent":
#             context["image"] = call_image_agent(image)
#         elif agent == "general":
#             context["general"] = call_general_agent(user_message)

#     context_prompt = "Voici les informations obtenues des différents agents :\n"
#     for key, value in context.items():
#         context_prompt += f"[{key.upper()}]: {value}\n"
#     context_prompt += "\nEn te basant uniquement sur ces informations, réponds à la question de l'utilisateur :\n"
#     context_prompt += user_message

#     response = LLM.chat.completions.create(
#         model=deployment,
#         stream=True,
#         messages=[
#             {"role": "system", "content": "Tu es un assistant intelligent expert dans la gestion de plante qui synthétise des réponses à partir des résultats d'autres agents."},
#             {"role": "user", "content": context_prompt}
#         ],
#         max_tokens=1024,
#         temperature=0.7
#     )

#     for update in response:
#         if update.choices:
#             print((update.choices[0].delta.content or "").encode('utf-8', errors='ignore').decode('utf-8'), end="")

#     # return response.choices[0].message.content
#     # return "End Message"

# if __name__ == "__main__":
#     message = input("Message: ")
#     orchestrate(message)