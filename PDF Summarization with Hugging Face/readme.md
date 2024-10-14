# PDF Summarization using Hugging Face

## Project Overview
In this project, we utilize Hugging Face's transformer models to summarize the content of PDF files. The pipeline includes:
1. Extracting text from PDF using PDF parsing libraries.
2. Summarizing the extracted text with a transformer-based model.
3. Outputting a concise summary.

The project leverages the pre-trained BART or T5 model from Hugging Face for the summarization task. These models are known for their performance in natural language understanding and generation tasks.

## Features
- Extract text from PDF files.
- Summarize long texts into concise summaries using transformer models.
- Easy to modify for different summarization use cases.

## Libraries
This project uses the following libraries:
- Hugging Face Transformers
- PyPDF2
- Torch
