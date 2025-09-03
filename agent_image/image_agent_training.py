import os
import time
from dotenv import load_dotenv
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry
from msrest.authentication import ApiKeyCredentials

# === Chargement des variables d'environnement ===
load_dotenv()

ENDPOINT = os.getenv("AZURE_AI_VISION_ENDPOINT")
TRAINING_KEY = os.getenv("AZURE_AI_VISION_TRAINING_KEY")
PROJECT_ID = os.getenv("AZURE_AI_VISION_PROJECT_ID")
PUBLISH_ITERATION_NAME = os.getenv("AZURE_AI_VISION_PUBLISH_ITERATION_NAME")
PREDICTION_RESOURCE_ID = os.getenv("AZURE_AI_VISION_RESOURCE_ID")

DATASET_PATH = "C:/Mickael/ITU/Stage DEEP IROULEGUY/AZURE/Agentic AI/Datasets/Grapevine"
TAG_NAMES = ["black_rot", "esca", "Healthy", "leaf_blight"]
MAX_IMAGES_PER_GROUP = 1000
BATCH_SIZE = 64


def authenticate():
    credentials = ApiKeyCredentials(in_headers={"Training-key": TRAINING_KEY})
    return CustomVisionTrainingClient(ENDPOINT, credentials)


def prepare_tags(trainer):
    existing_tags = {tag.name: tag for tag in trainer.get_tags(PROJECT_ID)}
    tags = {}
    for name in TAG_NAMES:
        if name in existing_tags:
            print(f"[INFO] Le tag '{name}' existe déjà.")
            tags[name] = existing_tags[name]
        else:
            print(f"[INFO] Création du tag '{name}'...")
            tags[name] = trainer.create_tag(PROJECT_ID, name)
    return tags


def upload_images(trainer, tags):
    for tag_name in TAG_NAMES:
        tag_id = tags[tag_name].id
        image_dir = os.path.join(DATASET_PATH, tag_name)
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        print(f"\n {len(image_files)} images à uploader pour le tag '{tag_name}'")

        for group_start in range(0, len(image_files), MAX_IMAGES_PER_GROUP):
            group_files = image_files[group_start:group_start + MAX_IMAGES_PER_GROUP]
            print(f"\n Envoi des images {group_start + 1} à {group_start + len(group_files)}...")

            image_batch = []
            for i, file_name in enumerate(group_files):
                image_path = os.path.join(image_dir, file_name)
                with open(image_path, "rb") as image_contents:
                    image_batch.append(
                        ImageFileCreateEntry(
                            name=file_name,
                            contents=image_contents.read(),
                            tag_ids=[tag_id]
                        )
                    )

                if len(image_batch) == BATCH_SIZE or i == len(group_files) - 1:
                    batch = ImageFileCreateBatch(images=image_batch)
                    result = trainer.create_images_from_files(PROJECT_ID, batch)

                    failed = any(img.status not in ("OK", "OKDuplicate") for img in result.images)
                    if failed:
                        print(f"[ERREUR] Upload échoué pour un batch (tag: {tag_name})")
                        for img in result.images:
                            print("   -", img.status)
                        exit(-1)
                    else:
                        print(f"[INFO] Batch de {len(image_batch)} images envoyé (OK ou doublons)")

                    image_batch = []

    print("\n Toutes les images ont été envoyées avec succès.")


def train_model(trainer):
    print("\n Lancement de l'entraînement du modèle...")
    iteration = trainer.train_project(PROJECT_ID)

    while iteration.status != "Completed":
        print(f"[INFO] Entraînement en cours : {iteration.status}...")
        time.sleep(10)
        iteration = trainer.get_iteration(PROJECT_ID, iteration.id)

    print(f"\n Entraînement terminé ! Iteration ID : {iteration.id}")
    return iteration


def publish_iteration(trainer, iteration):
    print(f"\n Publication de l’iteration '{PUBLISH_ITERATION_NAME}'...")
    trainer.publish_iteration(PROJECT_ID, iteration.id, PUBLISH_ITERATION_NAME, PREDICTION_RESOURCE_ID)
    print(f" Iteration publiée avec succès sous le nom : {PUBLISH_ITERATION_NAME}")


def publish_latest_completed_iteration(trainer):
    iterations = trainer.get_iterations(PROJECT_ID)
    latest_iteration = next((it for it in iterations if it.status == "Completed"), None)

    if latest_iteration:
        print(f"[INFO] Publication de l’itération '{latest_iteration.name}'...")
        trainer.publish_iteration(PROJECT_ID, latest_iteration.id, PUBLISH_ITERATION_NAME, PREDICTION_RESOURCE_ID)
        print(f"[SUCCÈS] Iteration publiée sous le nom : {PUBLISH_ITERATION_NAME}")
    else:
        print("[ERREUR] Aucune itération complétée disponible à publier.")


if __name__ == "__main__":
    trainer = authenticate()
    tags = prepare_tags(trainer)
    upload_images(trainer, tags)
    iteration = train_model(trainer)
    publish_iteration(trainer, iteration)
    # publish_latest_completed_iteration(trainer)










































# import os
# import time
# from dotenv import load_dotenv
# from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
# from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry
# from msrest.authentication import ApiKeyCredentials

# # === Chargement des variables d'environnement ===
# load_dotenv()

# ENDPOINT = os.getenv("AZURE_AI_VISION_ENDPOINT")
# TRAINING_KEY = os.getenv("AZURE_AI_VISION_TRAINING_KEY")
# PROJECT_ID = os.getenv("AZURE_AI_VISION_PROJECT_ID")
# PUBLISH_ITERATION_NAME = os.getenv("AZURE_AI_VISION_PUBLISH_ITERATION_NAME")
# PREDICTION_RESOURCE_ID = os.getenv("AZURE_AI_VISION_RESOURCE_ID")

# # === Authentification ===
# credentials = ApiKeyCredentials(in_headers={"Training-key": TRAINING_KEY})
# trainer = CustomVisionTrainingClient(ENDPOINT, credentials)

# # === Paramètres ===
# dataset_path = "C:/Mickael/ITU/Stage DEEP IROULEGUY/AZURE/Agentic AI/Datasets/Grapevine"
# tag_names = ["black_rot", "esca", "Healthy", "leaf_blight"]
# MAX_IMAGES_PER_GROUP = 1000
# BATCH_SIZE = 64

# # === Création ou récupération des tags ===
# existing_tags = {tag.name: tag for tag in trainer.get_tags(PROJECT_ID)}
# tags = {}

# for name in tag_names:
#     if name in existing_tags:
#         print(f"[INFO] Le tag '{name}' existe déjà.")
#         tags[name] = existing_tags[name]
#     else:
#         print(f"[INFO] Création du tag '{name}'...")
#         tags[name] = trainer.create_tag(PROJECT_ID, name)

# # === Upload des images ===
# for tag_name in tag_names:
#     tag_id = tags[tag_name].id
#     image_dir = os.path.join(dataset_path, tag_name)
#     image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

#     print(f"\n {len(image_files)} images à uploader pour le tag '{tag_name}'")

#     # Découpage en groupes de 1000 images
#     for group_start in range(0, len(image_files), MAX_IMAGES_PER_GROUP):
#         group_files = image_files[group_start:group_start + MAX_IMAGES_PER_GROUP]
#         print(f"\n Envoi des images {group_start + 1} à {group_start + len(group_files)}...")

#         image_batch = []

#         for i, file_name in enumerate(group_files):
#             image_path = os.path.join(image_dir, file_name)
#             with open(image_path, "rb") as image_contents:
#                 image_batch.append(
#                     ImageFileCreateEntry(
#                         name=file_name,
#                         contents=image_contents.read(),
#                         tag_ids=[tag_id]
#                     )
#                 )

#             # Envoi par batch de 64
#             if len(image_batch) == BATCH_SIZE or i == len(group_files) - 1:
#                 batch = ImageFileCreateBatch(images=image_batch)
#                 result = trainer.create_images_from_files(PROJECT_ID, batch)

#                 # Vérifie uniquement les vrais échecs (ignore OKDuplicate)
#                 failed = any(img.status not in ("OK", "OKDuplicate") for img in result.images)
#                 if failed:
#                     print(f"[ERREUR] Upload échoué pour un batch (tag: {tag_name})")
#                     for img in result.images:
#                         print("   -", img.status)
#                     exit(-1)
#                 else:
#                     print(f"[INFO] Batch de {len(image_batch)} images envoyé (OK ou doublons)")

#                 image_batch = []

# print("\n Toutes les images ont été envoyées avec succès.")

# # === Entraînement du modèle ===
# print("\n Lancement de l'entraînement du modèle...")
# iteration = trainer.train_project(PROJECT_ID)

# while iteration.status != "Completed":
#     print(f"[INFO] Entraînement en cours : {iteration.status}...")
#     time.sleep(10)
#     iteration = trainer.get_iteration(PROJECT_ID, iteration.id)

# print(f"\n Entraînement terminé ! Iteration ID : {iteration.id}")

# # === Publication de l'itération pour la prédiction ===
# print(f"\n Publication de l’iteration '{PUBLISH_ITERATION_NAME}'...")
# trainer.publish_iteration(PROJECT_ID, iteration.id, PUBLISH_ITERATION_NAME, PREDICTION_RESOURCE_ID)
# print(f" Iteration publiée avec succès sous le nom : {PUBLISH_ITERATION_NAME}")


# # Récupération de la dernière itération entraînée
# iterations = trainer.get_iterations(PROJECT_ID)
# latest_iteration = next((it for it in iterations if it.status == "Completed"), None)

# if latest_iteration:
#     print(f"[INFO] Publication de l’itération '{latest_iteration.name}'...")
#     trainer.publish_iteration(PROJECT_ID, latest_iteration.id, PUBLISH_ITERATION_NAME, PREDICTION_RESOURCE_ID)
#     print(f"[SUCCÈS] Iteration publiée sous le nom : {PUBLISH_ITERATION_NAME}")
# else:
#     print("[ERREUR] Aucune itération complétée disponible à publier.")