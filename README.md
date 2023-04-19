# OP_stack
Having some fun with OP stack (OpenAI + Pinecone)

## Architecture:
<img src="https://files.readme.io/6a3ea5a-pinecone-openai-overview.png">

### Pinecone Vector Index Specs:
- Cosine Metric
- 1536 dimensions (as per OpenAI's ADA embeddings model)
- s1.x1 pod type (optimized for storage, ~ 3,500,000 vectors)

### What to improve on:
- During ingestion, consider using `PDFMiner` from [LangChain's PDF Document Loader](https://python.langchain.com/en/latest/modules/indexes/document_loaders/examples/pdf.html#using-pdfminer-to-generate-html-text) that is able to generate HTML text. It will enrich the data with ability to parse font sizing, page numbers, headers/footers.

<!-- ## Installation:
1. Install `pinecone-client` (I used `--user` because had a permission error).
```
pip3 install pinecone-client --user
```
2. -->

## Resources Used during this project:
- [Pinecone documentation](https://docs.pinecone.io/docs/python-client)
- [OpenAI embeddings guide](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings)
- [LangChain documentation](https://python.langchain.com/en/latest/getting_started/getting_started.html)
- [Q&A Chat from PDF Architecture](https://www.youtube.com/watch?v=ih9PBGVVOO4&ab_channel=Chatwithdata)