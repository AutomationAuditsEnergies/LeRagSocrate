# nlp_service

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import logging
from datetime import datetime

# Configuration des logs pour Azure
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/tmp/rag_service.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# Ajouter le répertoire parent pour importer rag_agent
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from rag_agent import rag_answer, init_conversation_table

    logger.info("✅ Module rag_agent importé avec succès")
except ImportError as e:
    logger.error(f"❌ Erreur import rag_agent: {e}")
    raise

app = Flask(__name__)
CORS(app)  # Permettre les requêtes cross-origin depuis votre app principale

# Initialiser les tables au démarrage
try:
    init_conversation_table()
    logger.info("✅ Tables de conversation initialisées")
except Exception as e:
    logger.error(f"❌ Erreur initialisation tables: {e}")


@app.route("/", methods=["GET"])
def health_check():
    """Health check pour Azure"""
    try:
        return jsonify(
            {
                "status": "healthy",
                "service": "RAG Service (No ChromaDB)",
                "timestamp": datetime.now().isoformat(),
                "version": "2.0",
            }
        )
    except Exception as e:
        logger.error(f"❌ Erreur health check: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    """Endpoint principal - Compatible avec votre call_rag_service()"""
    try:
        data = request.json
        if not data:
            logger.warning("⚠️ Pas de données JSON reçues")
            return (
                jsonify(
                    {
                        "error": "Données JSON requises",
                        "answer_text": "Désolé, je n'ai pas reçu de données.",
                    }
                ),
                400,
            )

        question = data.get("question", "").strip()
        username = data.get("username", "utilisateur")  # Paramètre optionnel
        user_id = data.get("user_id")  # Paramètre optionnel

        if not question:
            logger.warning("⚠️ Question vide reçue")
            return (
                jsonify(
                    {
                        "error": "No question provided",
                        "answer_text": "Désolé, je n'ai pas reçu de question.",
                    }
                ),
                400,
            )

        logger.info(
            f"🔍 Microservice RAG: Question reçue de {username}: {question[:50]}..."
        )

        # Utiliser votre RAG avec les paramètres optionnels
        if username != "utilisateur" or user_id:
            # Si on a des infos utilisateur, les passer à rag_answer
            answer_text = rag_answer(question, username=username, user_id=user_id)
        else:
            # Utilisation simple comme avant
            answer_text = rag_answer(question)

        logger.info(
            f"✅ Microservice RAG: Réponse générée ({len(answer_text)} caractères)"
        )

        return jsonify(
            {
                "answer_text": answer_text,
                "question": question,
                "username": username,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"❌ Erreur microservice RAG: {e}")
        return (
            jsonify(
                {
                    "error": str(e),
                    "answer_text": "Désolé, Alain n'est pas disponible pour le moment.",
                }
            ),
            500,
        )


@app.route("/status", methods=["GET"])
def get_status():
    """Status détaillé du service - Version sans ChromaDB"""
    try:
        # Vérifier la base de données SQLite
        db_status = "unknown"
        try:
            import sqlite3

            conn = sqlite3.connect(os.getenv("DB_PATH", "/tmp/database.db"))
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM conversations")
            conv_count = cursor.fetchone()[0]
            conn.close()
            db_status = f"✅ {conv_count} conversations"
        except Exception as e:
            db_status = f"❌ {str(e)}"

        return jsonify(
            {
                "service": "RAG Microservice (No ChromaDB)",
                "status": "running",
                "database_status": db_status,
                "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
                "chromadb_status": "❌ Disabled (Azure compatibility)",
                "timestamp": datetime.now().isoformat(),
            }
        )
    except Exception as e:
        logger.error(f"❌ Erreur status: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    try:
        # Configuration pour Azure et local
        port = int(os.environ.get("PORT", 7000))
        debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() == "true"

        logger.info(f"🚀 Démarrage du microservice RAG sur le port {port}")
        logger.info(f"🔧 Mode debug: {debug_mode}")
        logger.info(
            f"🔐 OpenAI configuré: {'✅' if os.getenv('OPENAI_API_KEY') else '❌'}"
        )

        # En production Azure, utiliser 0.0.0.0
        host = "0.0.0.0" if os.getenv("PORT") else "localhost"

        app.run(host=host, port=port, debug=debug_mode)

    except Exception as e:
        logger.error(f"❌ Erreur critique au démarrage: {e}")
        raise
