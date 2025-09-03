from dotenv import load_dotenv
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials
import os, sys, time, uuid
sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

PREDICTION_ENDPOINT = os.getenv("AZURE_AI_VISION_PREDICTION_ENDPOINT")
PREDICTION_KEY = os.getenv("AZURE_AI_VISION_PREDICTION_KEY")
PROJECT_ID = os.getenv("AZURE_AI_VISION_PROJECT_ID")
PUBLISH_ITERATION_NAME = os.getenv("AZURE_AI_VISION_PUBLISH_ITERATION_NAME")

TRAINING_KEY = os.getenv("AZURE_AI_VISION_TRAINING_KEY")
TRAINING_ENDPOINT = os.getenv("AZURE_AI_VISION_ENDPOINT")

prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": PREDICTION_KEY})
predictor = CustomVisionPredictionClient(PREDICTION_ENDPOINT, prediction_credentials)

training_credentials = ApiKeyCredentials(in_headers={"Training-key": TRAINING_KEY})
trainer = CustomVisionTrainingClient(TRAINING_ENDPOINT, training_credentials)

def predict_image(image_path):
    with open(image_path, "rb") as image_data:
        results = predictor.classify_image(PROJECT_ID, PUBLISH_ITERATION_NAME, image_data.read())

    print(f"\nRésultats pour image : {os.path.basename(image_path)}")
    for prediction in results.predictions:
        print(f" - {prediction.tag_name} ({prediction.probability * 100:.2f}%)")

def check_project():
    projects = trainer.get_projects()
    project = next((p for p in projects if str(p.id) == PROJECT_ID), None)
    if not project:
        print(f"Le projet avec ID {PROJECT_ID} n'existe pas.")
        return False
    print(f" Projet trouvé : {project.name}")
    return True

def list_iterations():
    print(f"\nListe des itérations pour le projet {PROJECT_ID} :")
    iterations = trainer.get_iterations(PROJECT_ID)
    if not iterations:
        print("Aucune itération disponible.")
        return []
    for it in iterations:
        pub = f"(Publié sous '{it.publish_name}')" if it.publish_name else "(Non publié)"
        print(f" - {it.name} - Status: {it.status} {pub}")
    return iterations

def check_iteration_published():
    iterations = trainer.get_iterations(PROJECT_ID)
    for it in iterations:
        if it.publish_name == PUBLISH_ITERATION_NAME:
            print(f"itération '{it.name}' est bien publiée sous le nom : {PUBLISH_ITERATION_NAME}")
            return True
    print(f" Aucune itération publiée sous le nom : {PUBLISH_ITERATION_NAME}")
    return False

def test_azure_keys_and_endpoints():
    print(" Diagnostic de configuration Azure Custom Vision")
    ok_project = check_project()
    if not ok_project:
        exit(1)

    list_iterations()

    if not check_iteration_published():
        print("[] Impossible de prédire sans itération publiée. Corrige cela et relance.")
        exit(1)

    print("\nValidation des clés et endpoints Azure...")
    errors = 0

    try:
        train_cred = ApiKeyCredentials(in_headers={"Training-key": TRAINING_KEY})
        trainer_test = CustomVisionTrainingClient(TRAINING_ENDPOINT, train_cred)
        projects = trainer_test.get_projects()
        print(f"Clé d'entraînement VALIDE - {len(projects)} projet(s) accessible(s).")
    except Exception as e:
        print("Clé ou endpoint d'entraînement INVALIDE :", e)
        errors += 1

    try:
        pred_cred = ApiKeyCredentials(in_headers={"Prediction-key": PREDICTION_KEY})
        predictor_test = CustomVisionPredictionClient(PREDICTION_ENDPOINT, pred_cred)

        image_path = "C:\\Users\\16777\\Downloads\\test.jpg" 
        if not os.path.exists(image_path):
            print("Image de test manquante : crée un fichier 'test.jpg' pour ce test.")
        else:
            with open(image_path, "rb") as image_data:
                predictor_test.classify_image(PROJECT_ID, PUBLISH_ITERATION_NAME, image_data.read())
            print("Clé de prédiction VALIDE - prédiction test réussie.")
    except Exception as e:
        print("Clé ou endpoint de prédiction INVALIDE :", e)
        errors += 1

    try:
        trainer_check = CustomVisionTrainingClient(TRAINING_ENDPOINT, train_cred)
        iterations = trainer_check.get_iterations(PROJECT_ID)
        published = [it for it in iterations if it.publish_name == PUBLISH_ITERATION_NAME]
        if published:
            print(f"L’itération '{PUBLISH_ITERATION_NAME}' est bien publiée.")
        else:
            print(f" Aucune itération publiée sous le nom : {PUBLISH_ITERATION_NAME}")
            errors += 1
    except Exception as e:
        print(" Impossible de récupérer les itérations :", e)
        errors += 1

    print(f"\nRésumé : {3 - errors}/3 tests réussis.")
    if errors > 0:
        print("Des erreurs ont été détectées, merci de corriger les paramètres.")
    else:
        print("Tout est prêt pour la prédiction.")
    
def call_image_agent(image_path):
    """Appel pour l'agent orchestrateur – retourne les prédictions formatées"""
    try:
        if not os.path.exists(image_path):
            return {"error": f"Image introuvable à ce chemin : {image_path}"}

        with open(image_path, "rb") as image_data:
            results = predictor.classify_image(PROJECT_ID, PUBLISH_ITERATION_NAME, image_data.read())

        response = [
            {
                "tag_name": prediction.tag_name,
                "probability": round(prediction.probability * 100, 2)
            }
            for prediction in results.predictions
        ]
        return {"image": os.path.basename(image_path), "predictions": response}

    except Exception as e:
        return {"error": f"Erreur lors de la prédiction : {str(e)}"}


if __name__ == "__main__":
    # test_azure_keys_and_endpoints()

    image_to_test = "C:\\Users\\16777\\Downloads\\test.jpg"  # ← Remplace par ton chemin
    if os.path.exists(image_to_test):
        predict_image(image_to_test)
    else:
        print(f"[] Image introuvable : {image_to_test}")