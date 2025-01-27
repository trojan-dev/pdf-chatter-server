import express, { json } from 'express';
import multer, { memoryStorage } from 'multer';
import { createClient } from '@supabase/supabase-js';
import OpenAIApi from 'openai';
import { config } from 'dotenv';
import pdf from 'pdf-parse'
import cors from 'cors'

config();

const app = express();
const PORT = process.env.PORT || 5000;

// Supabase client
const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_ROLE_KEY);

// OpenAI client
const openai = new OpenAIApi({
    apiKey: process.env.OPENAI_API_KEY,
});

// Middleware
app.use(json());
app.use(cors());

const upload = multer({ storage: memoryStorage() });

app.post('/api/upload', upload.single('file'), async (req, res) => {
    const file = req.file;
  
    if (!file) {
      return res.status(400).json({ message: 'No file uploaded' });
    }
  
    if (file.mimetype !== 'application/pdf') {
      return res.status(400).json({ message: 'Invalid file type. Please upload a PDF.' });
    }
  
    try {
      // Step 1: Upload file to Supabase Storage
      const { data: uploadData, error: uploadError } = await supabase.storage
        .from('pdf-files') // Ensure this bucket exists in Supabase
        .upload(`uploads/${file.originalname}`, file.buffer);
  
      if (uploadError) {
        throw new Error(`Error uploading file to storage: ${uploadError.message}`);
      }
  
      const fileUrl = uploadData.path;
  
      // Step 2: Parse PDF text
      const pdfData = await pdf(file.buffer);
      const fullText = pdfData.text;
  
      // Step 3: Split PDF text into manageable chunks
      const chunkSize = 1000; // Customize chunk size (e.g., 1000 characters)
      const textChunks = [];
      for (let i = 0; i < fullText.length; i += chunkSize) {
        textChunks.push(fullText.slice(i, i + chunkSize));
      }
  
      // Step 4: Generate embeddings for each chunk
      const embeddingsResponse = await openai.embeddings.create({
        model: 'text-embedding-ada-002',
        input: textChunks,
      });
  
      const chunkEmbeddings = embeddingsResponse.data.map((d) => d.embedding);
  
      // Step 5: Save metadata, text_chunks, and embeddings to the database
      const { data: fileRecord, error: dbError } = await supabase
        .from('pdf_metadata')
        .insert({
          file_name: file.originalname,
          file_path: fileUrl,
          text_chunks: textChunks,
          embeddings: chunkEmbeddings,
        })
        .select();
  
      if (dbError) {
        throw new Error(`Error saving metadata to the database: ${dbError.message}`);
      }
  
      res.json({
        fileId: fileRecord[0].id,
        message: 'File uploaded and processed successfully',
      });
    } catch (error) {
      console.error('Error during file upload:', error);
      res.status(500).json({ message: 'Internal server error', error: error.message });
    }
});

const cosineSimilarity = (vecA, vecB) => {
    const dotProduct = vecA.reduce((sum, a, idx) => sum + a * vecB[idx], 0);
    const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
    const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
    return dotProduct / (magnitudeA * magnitudeB);
};
  
app.post('/api/chat', async (req, res) => {
    const { fileId, message } = req.body;

    if (!fileId || !message) {
    return res.status(400).json({ message: 'File ID and message are required.' });
    }

    try {
    // Fetch embeddings and associated metadata from Supabase
    const { data: fileRecord, error } = await supabase
        .from('pdf_metadata')
        .select('embeddings, file_name, text_chunks') // Assume text_chunks stores segmented text corresponding to embeddings
        .eq('id', fileId)
        .single();

    if (error || !fileRecord) {
        throw new Error('Error fetching embeddings or file not found.');
    }

    const { embeddings, file_name, text_chunks } = fileRecord;

    // Generate query embedding using OpenAI SDK
    const embeddingResponse = await openai.embeddings.create({
        model: 'text-embedding-ada-002',
        input: message,
    });

    const queryEmbedding = embeddingResponse.data[0].embedding;

    // Perform cosine similarity to find the most relevant chunk
    const similarities = embeddings.map((embedding, index) => ({
        index,
        similarity: cosineSimilarity(queryEmbedding, embedding),
    }));

    const mostRelevantChunkIndex = similarities.sort((a, b) => b.similarity - a.similarity)[0].index;
    const relevantText = text_chunks[mostRelevantChunkIndex];

    // Use OpenAI to generate a chat completion with relevant context
    const chatResponse = await openai.chat.completions.create({
        model: 'gpt-4',
        messages: [
        {
            role: 'system',
            content: `You are an assistant with access to the contents of the PDF titled "${file_name}". Use the provided context to answer questions.`,
        },
        {
            role: 'system',
            content: `Context: ${relevantText}`,
        },
        { role: 'user', content: message },
        ],
    });

    res.json({ answer: chatResponse.choices[0].message.content });
    } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Internal server error', error: error.message });
    }
});
  
  

// Start the server
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
