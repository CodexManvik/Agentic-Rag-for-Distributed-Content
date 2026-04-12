"""
ChromaDB to LanceDB migration script.

Migrates existing vector data from ChromaDB to LanceDB while preserving
metadata and knowledge base associations.
"""

import argparse
from pathlib import Path
from typing import Optional
import json

from loguru import logger

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.error("chromadb not installed. Cannot perform migration.")

from backend.app.vectorstore import LanceDBVectorStore


def migrate_chromadb_to_lancedb(
    chroma_path: Path,
    lance_path: Path,
    collection_name: str = "documents",
    batch_size: int = 100,
    dry_run: bool = False
):
    """
    Migrate data from ChromaDB to LanceDB.
    
    Args:
        chroma_path: Path to ChromaDB data directory
        lance_path: Path to LanceDB data directory
        collection_name: ChromaDB collection name
        batch_size: Number of documents to process per batch
        dry_run: If True, don't actually write to LanceDB
    """
    if not CHROMADB_AVAILABLE:
        raise RuntimeError("chromadb is required for migration")
    
    logger.info(f"Starting migration from {chroma_path} to {lance_path}")
    
    # Connect to ChromaDB
    logger.info(f"Connecting to ChromaDB at {chroma_path}")
    chroma_client = chromadb.PersistentClient(path=str(chroma_path))
    
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except Exception as e:
        logger.error(f"Failed to get collection '{collection_name}': {e}")
        logger.info(f"Available collections: {chroma_client.list_collections()}")
        raise
    
    # Get all documents
    logger.info(f"Reading documents from ChromaDB collection '{collection_name}'")
    
    try:
        # ChromaDB get() with no arguments returns all documents
        results = collection.get(
            include=["documents", "metadatas", "embeddings"]
        )
        
        total_docs = len(results['ids'])
        logger.info(f"Found {total_docs} documents in ChromaDB")
        
        if total_docs == 0:
            logger.warning("No documents to migrate")
            return
        
    except Exception as e:
        logger.error(f"Failed to read from ChromaDB: {e}")
        raise
    
    if dry_run:
        logger.info("Dry run mode - not writing to LanceDB")
        logger.info(f"Would migrate {total_docs} documents")
        return
    
    # Initialize LanceDB
    logger.info(f"Initializing LanceDB at {lance_path}")
    lance_store = LanceDBVectorStore(
        db_path=lance_path,
        table_name="documents"
    )
    
    # Migrate in batches
    migrated_count = 0
    
    for i in range(0, total_docs, batch_size):
        batch_end = min(i + batch_size, total_docs)
        
        logger.info(f"Migrating batch {i//batch_size + 1} "
                   f"(documents {i+1}-{batch_end} of {total_docs})")
        
        # Prepare batch grouped by knowledge base to preserve per-document KB routing.
        grouped_docs: dict[str, list[dict[str, object]]] = {}
        grouped_embeddings: dict[str, list[list[float]]] = {}
        
        for j in range(i, batch_end):
            # Extract knowledge base from metadata or use default
            metadata = results['metadatas'][j] or {}
            kb = metadata.get('knowledge_base', 'default')
            
            doc = {
                'text': results['documents'][j],
                'metadata': metadata
            }
            
            grouped_docs.setdefault(kb, []).append(doc)
            grouped_embeddings.setdefault(kb, []).append(results['embeddings'][j])
        
        # Add to LanceDB
        try:
            inserted_in_batch = 0
            for kb, docs_for_kb in grouped_docs.items():
                embeddings_for_kb = grouped_embeddings.get(kb, [])
                lance_store.add_documents(
                    documents=docs_for_kb,
                    embeddings=embeddings_for_kb,
                    knowledge_base=kb
                )
                inserted_in_batch += len(docs_for_kb)

            migrated_count += inserted_in_batch
            logger.info(f"Migrated {migrated_count}/{total_docs} documents")
            
        except Exception as e:
            logger.error(f"Failed to migrate batch: {e}")
            raise
    
    logger.info(f"Migration complete! Migrated {migrated_count} documents")
    
    # Verify migration
    stats = lance_store.get_stats()
    logger.info(f"LanceDB stats: {stats}")


def main():
    """CLI entry point for migration script."""
    parser = argparse.ArgumentParser(
        description="Migrate ChromaDB vector store to LanceDB"
    )
    
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=Path("chroma_data"),
        help="Path to ChromaDB data directory (default: chroma_data)"
    )
    
    parser.add_argument(
        "--lance-path",
        type=Path,
        default=Path("lance_data"),
        help="Path to LanceDB data directory (default: lance_data)"
    )
    
    parser.add_argument(
        "--collection",
        type=str,
        default="documents",
        help="ChromaDB collection name (default: documents)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for migration (default: 100)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without writing to LanceDB"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<level>{message}</level>",
        level="INFO"
    )
    
    try:
        migrate_chromadb_to_lancedb(
            chroma_path=args.chroma_path,
            lance_path=args.lance_path,
            collection_name=args.collection,
            batch_size=args.batch_size,
            dry_run=args.dry_run
        )
        
        if not args.dry_run:
            logger.success("✓ Migration completed successfully")
            logger.info("\nNext steps:")
            logger.info("1. Verify data in LanceDB")
            logger.info("2. Update application config to use LanceDB")
            logger.info("3. Test application with new vector store")
            logger.info("4. Backup and remove ChromaDB data once verified")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
