# rag_agent.py

import os
import time
import random
import sqlite3
from datetime import datetime
import chromadb
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import logging

# Configuration du logger
logger = logging.getLogger(__name__)

# Configuration depuis variables d'environnement (Azure friendly)
SECRET_KEY = os.getenv("OPENAI_API_KEY")
DB_PATH = os.getenv("DB_PATH", "/tmp/database.db")  # Azure utilise /tmp
CHROMA_PATH = os.getenv("CHROMA_PATH", "/tmp/chroma_simple")  # Azure utilise /tmp

logger.info(f"🔧 RAG Agent configuré - DB: {DB_PATH}, ChromaDB: {CHROMA_PATH}")

# Prompt pour guider l'agent avec mémoire
prompt_template = PromptTemplate(
    input_variables=["context", "conversation_history", "question"],
    template="""
    Tu es un formateur expérimenté et passionné qui accompagne des conseillers relation client à distance. 
    Tu as 15 ans d'expérience dans le domaine et tu adores transmettre ton savoir.
    
    Tu tutoies toujours tes apprenants et tu es bienveillant, encourageant, et parfois un peu taquin (de manière positive).
    Tu utilises des expressions naturelles comme "Alors", "Écoute", "Tu vois", "D'ailleurs", "Au fait".
    Tu donnes des exemples concrets et des anecdotes quand c'est pertinent.
    
    IMPORTANT: 
    - Utilise l'historique de conversation pour comprendre le contexte
    - Si l'apprenant répond par "oui", "non", "ok", fais référence à votre discussion précédente
    - Sois naturel et spontané, pas robotique
    - N'utilise JAMAIS d'émojis dans tes réponses
    - Termine parfois par une question pour relancer la conversation

    Historique de la conversation récente :
    {conversation_history}

    Contenu du cours disponible :
    {context}

    Question actuelle de l'apprenant :
    {question}

    Réponse d'Alain :
    """,
)

# Variables globales pour ChromaDB et LLM (initialisées à la demande)
client = None
collection = None
llm = None


def initialize_chromadb():
    """Initialise ChromaDB de manière lazy"""
    global client, collection

    if client is not None:
        return  # Déjà initialisé

    try:
        logger.info(f"🔗 Connexion à ChromaDB: {CHROMA_PATH}")

        # Créer le dossier si nécessaire
        os.makedirs(CHROMA_PATH, exist_ok=True)

        client = chromadb.PersistentClient(path=CHROMA_PATH)

        try:
            collection = client.get_collection(name="documents")
            doc_count = collection.count()
            logger.info(f"✅ Collection 'documents' trouvée: {doc_count} documents")
        except Exception:
            logger.warning("⚠️ Collection 'documents' non trouvée, création...")
            collection = client.create_collection(name="documents")
            logger.info("✅ Collection 'documents' créée")

    except Exception as e:
        logger.error(f"❌ Erreur connexion ChromaDB: {e}")
        # En cas d'erreur, on continue sans ChromaDB
        client = None
        collection = None


def initialize_llm():
    """Initialise le LLM de manière lazy"""
    global llm

    if llm is not None:
        return  # Déjà initialisé

    try:
        if SECRET_KEY:
            logger.info("🤖 Configuration OpenAI GPT-4")
            llm = ChatOpenAI(
                model_name="gpt-4",
                temperature=0.8,
                openai_api_key=SECRET_KEY,
                max_tokens=800,
            )
            logger.info("✅ LLM OpenAI configuré")
        else:
            logger.warning("⚠️ Pas de clé OpenAI - réponses simplifiées")
            llm = None

    except Exception as e:
        logger.error(f"❌ Erreur configuration LLM: {e}")
        llm = None


def init_conversation_table():
    """Initialise la table des conversations si elle n'existe pas"""
    try:
        # Créer le dossier parent si nécessaire
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                username TEXT,
                role TEXT,
                message TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                session_id TEXT,
                FOREIGN KEY (user_id) REFERENCES logs(id)
            )
        """
        )

        # Index pour optimiser les requêtes
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_conversations_user_time 
            ON conversations(username, timestamp DESC)
        """
        )

        conn.commit()
        conn.close()
        logger.info("✅ Table conversations initialisée")

    except Exception as e:
        logger.error(f"❌ Erreur initialisation table: {e}")
        # En cas d'erreur, on continue (la BDD sera créée plus tard)


def get_conversation_history(username: str, limit: int = 6) -> str:
    """Récupère l'historique récent de conversation depuis la BDD"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Récupérer les derniers messages pour cet utilisateur
        cursor.execute(
            """
            SELECT role, message, timestamp 
            FROM conversations 
            WHERE username = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        """,
            (username, limit),
        )

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return "Début de la conversation."

        # Inverser l'ordre pour avoir chronologique
        rows.reverse()

        formatted_history = []
        for role, message, timestamp in rows:
            # Raccourcir les messages longs pour le contexte
            short_message = message[:150] + "..." if len(message) > 150 else message
            formatted_history.append(f"{role}: {short_message}")

        return "\n".join(formatted_history)

    except Exception as e:
        logger.error(f"❌ Erreur récupération historique: {e}")
        return "Début de la conversation."


def add_to_conversation(username: str, role: str, message: str, user_id: int = None):
    """Ajoute un message à l'historique de conversation en BDD"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO conversations (user_id, username, role, message) 
            VALUES (?, ?, ?, ?)
        """,
            (user_id, username, role, message),
        )

        conn.commit()
        conn.close()

        logger.debug(f"💾 Message sauvé: {username} ({role})")

    except Exception as e:
        logger.error(f"❌ Erreur sauvegarde conversation: {e}")


def get_user_stats(username: str) -> dict:
    """Récupère des stats sur l'utilisateur (questions posées, etc.)"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT 
                COUNT(*) as total_messages,
                COUNT(CASE WHEN role = 'Utilisateur' THEN 1 END) as questions_posees,
                MIN(timestamp) as premiere_interaction,
                MAX(timestamp) as derniere_interaction
            FROM conversations 
            WHERE username = ?
        """,
            (username,),
        )

        row = cursor.fetchone()
        conn.close()

        return {
            "total_messages": row[0] or 0,
            "questions_posees": row[1] or 0,
            "premiere_interaction": row[2],
            "derniere_interaction": row[3],
        }

    except Exception as e:
        logger.error(f"❌ Erreur stats utilisateur: {e}")
        return {"total_messages": 0, "questions_posees": 0}


def cleanup_old_conversations(days_old: int = 30):
    """Nettoie les anciennes conversations (optionnel)"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(
            """
            DELETE FROM conversations 
            WHERE timestamp < datetime('now', '-{} days')
        """.format(
                days_old
            )
        )

        deleted = cursor.rowcount
        conn.commit()
        conn.close()

        logger.info(f"🧹 {deleted} anciennes conversations supprimées")

    except Exception as e:
        logger.error(f"❌ Erreur nettoyage conversations: {e}")


def rag_answer(
    question: str, username: str = "utilisateur", user_id: int = None
) -> str:
    """Fonction principale RAG - Compatible avec votre utilisation existante"""
    try:
        # Initialisation lazy des composants
        initialize_chromadb()
        initialize_llm()

        logger.info(f"🔍 Recherche pour {username}: {question[:50]}...")

        # Récupérer l'historique de conversation depuis la BDD
        conversation_history = get_conversation_history(username)

        # Ajouter la question de l'utilisateur à la BDD
        add_to_conversation(username, "Utilisateur", question, user_id)

        # Recherche dans ChromaDB
        context = ""
        if collection is not None:
            try:
                results = collection.query(
                    query_texts=[question],
                    n_results=8,
                    include=["documents", "distances"],
                )

                # Extraire le contexte
                if results["documents"] and results["documents"][0]:
                    context = "\n\n".join(results["documents"][0])
                    logger.debug(f"📚 Contexte trouvé: {len(context)} caractères")
                else:
                    context = "Aucun contenu spécifique trouvé dans le cours."

            except Exception as e:
                logger.error(f"❌ Erreur recherche ChromaDB: {e}")
                context = "Erreur lors de la recherche dans les documents."
        else:
            context = "Base de données vectorielle non disponible."

        # Générer la réponse
        if llm and SECRET_KEY:
            # Utiliser OpenAI GPT-4
            try:
                formatted_prompt = prompt_template.format(
                    context=context,
                    conversation_history=conversation_history,
                    question=question,
                )

                response = llm.invoke(formatted_prompt)

                # Extraire le texte de la réponse
                if hasattr(response, "content"):
                    result = response.content
                else:
                    result = str(response)

                logger.info(f"✅ Réponse GPT-4 générée: {len(result)} caractères")

            except Exception as e:
                logger.error(f"❌ Erreur GPT-4: {e}")
                result = f"Désolé, j'ai un problème technique. Voici ce que j'ai trouvé dans le cours :\n\n{context[:500] if context else 'Aucune information disponible.'}..."
        else:
            # Réponse simple sans LLM (fallback)
            if (
                context
                and "Aucun contenu spécifique" not in context
                and "Erreur" not in context
            ):
                result = f"D'après le contenu du cours, voici les informations pertinentes :\n\n{context[:600]}..."
            else:
                result = "Désolé, je n'ai pas trouvé d'informations pertinentes dans le cours pour répondre à ta question. Assure-toi que la base de données est bien configurée."

            logger.info("✅ Réponse fallback générée (sans LLM)")

        # Ajouter la réponse d'Alain à la BDD
        add_to_conversation(username, "Alain", result, user_id)

        # Simulation d'un temps de réponse humain (réduit pour Azure)
        if not os.getenv("AZURE_ENV") and not os.getenv("PORT"):  # Seulement en local
            base_delay = random.uniform(0.8, 2.5)
            length_delay = len(result) * random.uniform(0.015, 0.025)
            thinking_delay = random.uniform(0.5, 1.2)
            total_delay = min(base_delay + length_delay + thinking_delay, 6.0)

            logger.debug(f"💭 Alain réfléchit... ({thinking_delay:.1f}s)")
            time.sleep(total_delay)

        logger.info(f"✅ Réponse générée: {len(result)} caractères")
        return result

    except Exception as e:
        logger.error(f"❌ Erreur RAG complète: {e}")
        return "Je ne peux pas répondre à ta question pour le moment. Une erreur technique est survenue."


# Initialiser la table au démarrage du module
try:
    init_conversation_table()
except Exception as e:
    logger.error(f"❌ Erreur initialisation module: {e}")

# Test rapide (seulement si exécuté directement)
if __name__ == "__main__":
    logger.info("🧪 Test du RAG ChromaDB avec BDD...")

    username = "TestUser"

    # Test conversation
    response1 = rag_answer("Qu'est-ce que la motivation hédoniste ?", username)
    print(f"\n📝 Q1: Qu'est-ce que la motivation hédoniste ?")
    print(f"🤖 R1: {response1}")

    response2 = rag_answer("oui", username)
    print(f"\n📝 Q2: oui")
    print(f"🤖 R2: {response2}")

    # Afficher les stats
    stats = get_user_stats(username)
    print(f"\n📊 Stats pour {username}: {stats}")

    # Afficher l'historique
    print(f"\n🧠 Historique:")
    print(get_conversation_history(username, 10))
