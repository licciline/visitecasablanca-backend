const express = require('express');
const cors = require('cors');
const { OpenAI } = require('openai');
const { Pinecone } = require('@pinecone-database/pinecone');

const app = express();
app.use(cors());
app.use(express.json());

// Initialisation des configurations avec vos variables d'environnement
const openai = new OpenAI({
apiKey: process.env.OPENAI_API_KEY,
});

const pc = new Pinecone({
apiKey: process.env.PINECONE_API_KEY,
});
const indexName = process.env.PINECONE_INDEX_NAME || 'casablanca-knowledge';

// Route principale pour interroger l'IA (Utilisée par api.visitecasablanca.com)
app.post('/api/chat', async (req, res) => {
try {
const { message, history = [] } = req.body;

if (!message) {
return res.status(400).json({ error: "Le message est requis." });
}

// 1. Génération de l'embedding pour la recherche sémantique
const embeddingResponse = await openai.embeddings.create({
model: "text-embedding-3-small",
input: message,
encoding_format: "float",
});
const queryVector = embeddingResponse.data[0].embedding;

// 2. Recherche des données locales dans la mémoire Pinecone
const index = pc.index(indexName);
const queryResponse = await index.query({
vector: queryVector,
topK: 5,
includeMetadata: true,
});

// Extraction du contexte trouvé pour Casablanca
const localContext = queryResponse.matches
.map(match => (match.metadata && match.metadata.text) || '')
.filter(text => text !== '')
.join('\n\n');

// 3. Construction du prompt pour l'IA
const systemPrompt = `Tu es l'assistant IA officiel du projet VISITECASABLANCA (visitecasablanca.com).
Tu es un guide expert de la ville de Casablanca. Tu aides les utilisateurs à trouver des entreprises, des commerces, des importateurs, des activités pour enfants, et des lieux d'intérêt.
Tu dois répondre chaleureusement, de préférence en Darija marocain (ou en français si l'utilisateur te parle en français).

Voici les données réelles et locales extraites de notre base de données pour t'aider à répondre précisément :
\"\"\"
${localContext}
\"\"\"

Sers-toi de ce contexte pour donner des réponses précises (adresses, spécialités). Si la réponse ne s'y trouve pas, utilise tes connaissances générales sur Casablanca pour rester le plus utile possible.`;

// 4. Appel au modèle OpenAI GPT-4o-mini
const chatCompletion = await openai.chat.completions.create({
model: "gpt-4o-mini",
messages: [
{ role: "system", content: systemPrompt },
...history,
{ role: "user", content: message }
],
temperature: 0.7,
});

const reply = chatCompletion.choices[0].message.content;
res.json({ reply });

} catch (error) {
console.error("Erreur Serveur IA:", error);
res.status(500).json({ error: "Une erreur est survenue lors du traitement de la demande." });
}
});

// Route de diagnostic
app.get('/', (req, res) => {
res.send("Le serveur VISITECASABLANCA AI est en ligne et fonctionnel !");
});

const PORT = process.env.PORT || 10000;
app.listen(PORT, () => {
console.log(`Serveur démarré sur le port ${PORT}`);
});
