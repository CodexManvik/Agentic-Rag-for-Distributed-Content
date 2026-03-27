# Ingestion Report

Run timestamp: 2026-03-27T05:43:27Z
Resource pack: hackathon_public_pack_v1
Resource pack path: backend\resources\resource_pack.yaml
Total duration seconds: 107.01

## Summary

- documents_processed: 32
- chunks_added: 1284
- skipped_duplicates: 30
- success_count: 30
- failed_count: 2

## Processed URLs
- https://support.atlassian.com/confluence-cloud/docs/make-a-space-public/
- https://support.atlassian.com/confluence-cloud/docs/set-up-and-manage-public-links/
- https://support.atlassian.com/confluence-cloud/docs/manage-public-links-across-confluence-cloud/
- https://support.atlassian.com/confluence-cloud/docs/how-secure-are-public-links/
- https://confluence.atlassian.com/doc/spaces-139459.html
- https://confluence.atlassian.com/doc/create-and-edit-pages-139459.html
- https://confluence.atlassian.com/doc/use-confluence-cloud-139459.html
- https://confluence.atlassian.com/doc/confluence-user-s-guide-139764.html
- https://confluence.atlassian.com/doc/search-139678.html
- https://docs.langchain.com/oss/python/langchain/rag
- https://docs.langchain.com/oss/python/langchain/retrieval
- https://python.langchain.com/docs/tutorials/rag/
- https://python.langchain.com/docs/concepts/
- https://python.langchain.com/docs/introduction/
- https://www.atlassian.com/software/confluence/demo
- https://www.langchain.com/retrieval
- https://langchain-ai.github.io/langgraph/concepts/why-langgraph/
- https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/
- https://langchain-ai.github.io/langgraph/concepts/low_level/
- https://docs.langchain.com/oss/python/langchain/overview
- https://ai.google.dev/gemini-api/docs
- https://www.anthropic.com/engineering

## Processed PDFs
- https://arxiv.org/pdf/2005.11401.pdf
- https://arxiv.org/pdf/2312.10997.pdf
- https://arxiv.org/pdf/2310.11511.pdf
- https://arxiv.org/pdf/2309.07864.pdf
- https://arxiv.org/pdf/2201.11903.pdf
- https://arxiv.org/pdf/1706.03762.pdf
- https://arxiv.org/pdf/2307.03172.pdf
- https://arxiv.org/pdf/2401.18059.pdf
- https://arxiv.org/pdf/2404.07143.pdf
- https://arxiv.org/pdf/2402.05102.pdf

## Errors
- https://confluence.atlassian.com/doc/confluence-user-s-guide-139764.html: 404 Client Error: Not Found for url: https://confluence.atlassian.com/doc/confluence-user-s-guide-139764.html
- https://confluence.atlassian.com/doc/search-139678.html: 404 Client Error: Not Found for url: https://confluence.atlassian.com/doc/search-139678.html

## Source Results
- web | ok | support.atlassian.com | chunks=9 | source=https://support.atlassian.com/confluence-cloud/docs/make-a-space-public/
- web | ok | support.atlassian.com | chunks=13 | source=https://support.atlassian.com/confluence-cloud/docs/set-up-and-manage-public-links/
- web | ok | support.atlassian.com | chunks=15 | source=https://support.atlassian.com/confluence-cloud/docs/manage-public-links-across-confluence-cloud/
- web | ok | support.atlassian.com | chunks=25 | source=https://support.atlassian.com/confluence-cloud/docs/how-secure-are-public-links/
- web | ok | confluence.atlassian.com | chunks=5 | source=https://confluence.atlassian.com/doc/spaces-139459.html
- web | ok | confluence.atlassian.com | chunks=0 | source=https://confluence.atlassian.com/doc/create-and-edit-pages-139459.html
- web | ok | confluence.atlassian.com | chunks=0 | source=https://confluence.atlassian.com/doc/use-confluence-cloud-139459.html
- web | failed | confluence.atlassian.com | chunks=0 | source=https://confluence.atlassian.com/doc/confluence-user-s-guide-139764.html
  error: 404 Client Error: Not Found for url: https://confluence.atlassian.com/doc/confluence-user-s-guide-139764.html
- web | failed | confluence.atlassian.com | chunks=0 | source=https://confluence.atlassian.com/doc/search-139678.html
  error: 404 Client Error: Not Found for url: https://confluence.atlassian.com/doc/search-139678.html
- web | ok | docs.langchain.com | chunks=14 | source=https://docs.langchain.com/oss/python/langchain/rag
- web | ok | docs.langchain.com | chunks=6 | source=https://docs.langchain.com/oss/python/langchain/retrieval
- web | ok | python.langchain.com | chunks=0 | source=https://python.langchain.com/docs/tutorials/rag/
- web | ok | python.langchain.com | chunks=1 | source=https://python.langchain.com/docs/concepts/
- web | ok | python.langchain.com | chunks=0 | source=https://python.langchain.com/docs/introduction/
- web | ok | www.atlassian.com | chunks=4 | source=https://www.atlassian.com/software/confluence/demo
- web | ok | www.langchain.com | chunks=5 | source=https://www.langchain.com/retrieval
- web | ok | langchain-ai.github.io | chunks=0 | source=https://langchain-ai.github.io/langgraph/concepts/why-langgraph/
- web | ok | langchain-ai.github.io | chunks=0 | source=https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/
- web | ok | langchain-ai.github.io | chunks=0 | source=https://langchain-ai.github.io/langgraph/concepts/low_level/
- web | ok | docs.langchain.com | chunks=0 | source=https://docs.langchain.com/oss/python/langchain/overview
- web | ok | ai.google.dev | chunks=6 | source=https://ai.google.dev/gemini-api/docs
- web | ok | www.anthropic.com | chunks=2 | source=https://www.anthropic.com/engineering
- pdf_url | ok | arxiv.org | chunks=73 | source=https://arxiv.org/pdf/2005.11401.pdf
- pdf_url | ok | arxiv.org | chunks=116 | source=https://arxiv.org/pdf/2312.10997.pdf
- pdf_url | ok | arxiv.org | chunks=113 | source=https://arxiv.org/pdf/2310.11511.pdf
- pdf_url | ok | arxiv.org | chunks=375 | source=https://arxiv.org/pdf/2309.07864.pdf
- pdf_url | ok | arxiv.org | chunks=151 | source=https://arxiv.org/pdf/2201.11903.pdf
- pdf_url | ok | arxiv.org | chunks=43 | source=https://arxiv.org/pdf/1706.03762.pdf
- pdf_url | ok | arxiv.org | chunks=69 | source=https://arxiv.org/pdf/2307.03172.pdf
- pdf_url | ok | arxiv.org | chunks=82 | source=https://arxiv.org/pdf/2401.18059.pdf
- pdf_url | ok | arxiv.org | chunks=51 | source=https://arxiv.org/pdf/2404.07143.pdf
- pdf_url | ok | arxiv.org | chunks=106 | source=https://arxiv.org/pdf/2402.05102.pdf
