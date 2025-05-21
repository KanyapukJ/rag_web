import os
from typing import List, Dict, Any
from datetime import datetime
import asyncio

from langchain_ollama import OllamaEmbeddings, OllamaLLM

# Initialize Ollama models
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")

# Initialize LLM with streaming capability
llm = OllamaLLM(
    model=LLM_MODEL,
    base_url=OLLAMA_HOST
)

# Initialize embeddings
embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_HOST
)

async def get_db_stats(collection):
    """Get statistics about the current ChromaDB collection asynchronously."""
    try:
        loop = asyncio.get_event_loop()
        # Assuming collection.get() is a synchronous (blocking) call
        results = await loop.run_in_executor(None, lambda: collection.get(include=["metadatas"]))

        if not results["metadatas"]:
            return None

        # Get unique URLs
        urls = set(meta["url"] for meta in results["metadatas"])

        # Get domains/sources
        domains = set(meta["source"] for meta in results["metadatas"])

        # Get document count
        doc_count = len(results["ids"])

        # Format last updated time
        last_updated = max(meta.get("crawled_at", "") for meta in results["metadatas"])
        if last_updated:
            # Convert to local timezone
            dt = datetime.fromisoformat(last_updated.replace("Z", "+00:00"))
            local_tz = datetime.now().astimezone().tzinfo
            dt = dt.astimezone(local_tz)
            last_updated = dt.strftime("%Y-%m-%d %H:%M:%S %Z")

        return {
            "urls": list(urls),
            "domains": list(domains),
            "doc_count": doc_count,
            "last_updated": last_updated,
        }
    except Exception as e:
        print(f"Error getting DB stats: {e}")
        return None


async def query_rag_system(collection, query: str, chat_history: List[Dict[str, str]] = [], num_results: int = 3) -> Dict[str, Any]:
    """Query the RAG system with a user question and chat history."""
    try:
        query_embedding = await embeddings.aembed_query(query)

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, # Use default ThreadPoolExecutor
            lambda: collection.query(
                query_embeddings=[query_embedding],
                n_results=num_results,
                include=["documents", "metadatas"],
            )
        )

        context_parts = []
        sources = []
        if results["documents"] and results["documents"][0]:
            for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
                context_parts.append(f"- {metadata.get('title', 'ข้อมูลอ้างอิง')}: {doc}")
                sources.append({
                    "title": metadata.get("title", "Unknown title"),
                    "url": metadata.get("url", "#"),
                    "summary": metadata.get("summary", "No summary available")
                })
        
        context = "\\n".join(context_parts)

        # System prompt (already updated to be in Thai and detailed)
        system_prompt = """คุณคือผู้ช่วย AI อัจฉริยะของ Agnos Health ที่เชี่ยวชาญด้านการให้ข้อมูลสุขภาพจากกระทู้ถามตอบทางการแพทย์
คุณได้รับคำถามจากผู้ใช้ และมีข้อมูลอ้างอิงจากกระทู้ที่แพทย์ได้ตอบไว้ในฟอรั่มของ Agnos Health

**คำแนะนำสำคัญ:**
1.  **ตอบเป็นภาษาไทยเสมอ** ตามภาษาของคำถามของผู้ใช้
2.  **พิจารณาประวัติการสนทนาก่อนหน้า (ถ้ามี) เพื่อทำความเข้าใจบริบทของคำถามปัจจุบัน**
3.  **ใช้ข้อมูลจาก "เนื้อหาอ้างอิง" ที่ให้มาเท่านั้น** ในการตอบคำถามปัจจุบัน ห้ามใช้ความรู้ส่วนตัวหรือข้อมูลจากภายนอกโดยเด็ดขาด
4.  **สวมบทบาทเป็นผู้ช่วยที่ให้ข้อมูล (ไม่ใช่แพทย์):** สรุปคำตอบจากแพทย์ในเนื้อหาอ้างอิงอย่างถูกต้องและเข้าใจง่าย
5.  **หากมีหลายคำตอบจากแพทย์ที่เกี่ยวข้อง:** พยายามสังเคราะห์ข้อมูลเหล่านั้นเพื่อให้คำตอบที่ครอบคลุมที่สุด ถ้าคำตอบขัดแย้งกัน ให้ระบุว่ามีความเห็นที่แตกต่างกันในเนื้อหาอ้างอิง
6.  **เน้นความแม่นยำ:** ตอบให้ตรงประเด็น กระชับ และเป็นข้อเท็จจริงตามเนื้อหาอ้างอิง
7.  **ถ้าข้อมูลไม่เพียงพอ:** หากเนื้อหาอ้างอิงไม่เกี่ยวข้องกับคำถามปัจจุบัน หรือไม่มีข้อมูลเพียงพอที่จะตอบได้อย่างมั่นใจ ให้ตอบว่า "ขออภัยค่ะ/ครับ ดิฉันไม่พบข้อมูลที่เกี่ยวข้องโดยตรงกับคำถามของคุณในฐานข้อมูล ณ ขณะนี้" หรือ "ข้อมูลที่มีอยู่อาจไม่เพียงพอที่จะให้คำตอบที่ชัดเจนสำหรับคำถามนี้ค่ะ/ครับ"
8.  **ความปลอดภัยและการปฏิเสธความรับผิดชอบ:**
    *   **ทุกครั้งที่ตอบคำถาม** ให้ปิดท้ายคำตอบด้วยข้อความนี้เสมอ: "ข้อมูลนี้เป็นเพียงข้อมูลเบื้องต้นที่สรุปจากกระทู้ถามตอบในฟอรั่ม Agnos Health และไม่สามารถใช้แทนคำแนะนำ การวินิจฉัย หรือการรักษาจากแพทย์ผู้เชี่ยวชาญได้ หากคุณมีข้อกังวลด้านสุขภาพ กรุณาปรึกษาแพทย์โดยตรงนะคะ/ครับ"
    *   ห้ามให้คำแนะนำทางการแพทย์ส่วนบุคคล ห้ามวินิจฉัยโรค หรือแนะนำการรักษาที่เฉพาะเจาะจง
9.  **อย่าอ้างอิงถึง "เนื้อหาอ้างอิง" หรือ "ประวัติการสนทนา" โดยตรง** ในคำตอบของคุณ แต่ให้ตอบเหมือนคุณมีความรู้นั้นอยู่แล้ว โดยไม่มีคำว่า 'ประวัติการสนทนา'

**รูปแบบการตอบ:**
[คำตอบที่สรุปจากเนื้อหาอ้างอิง โดยพิจารณาบริบทจากประวัติการสนทนา (ถ้ามี)]

ข้อมูลนี้เป็นเพียงข้อมูลเบื้องต้นที่สรุปจากกระทู้ถามตอบในฟอรั่ม Agnos Health และไม่สามารถใช้แทนคำแนะนำ การวินิจฉัย หรือการรักษาจากแพทย์ผู้เชี่ยวชาญได้ หากคุณมีข้อกังวลด้านสุขภาพ กรุณาปรึกษาแพทย์โดยตรงนะคะ/ครับ
"""

        # Format chat history for the prompt
        formatted_chat_history = ""
        if chat_history:
            history_lines = ["ประวัติการสนทนาก่อนหน้า:"]
            for msg in chat_history:
                role = "ผู้ใช้" if msg["role"] == "user" else "ผู้ช่วย AI"
                history_lines.append(f"{role}: {msg['content']}")
            formatted_chat_history = "\\n".join(history_lines) + "\\n\\n"
        
        # Construct the final prompt
        prompt_parts = [
            f"System: {system_prompt}",
            formatted_chat_history + f"เนื้อหาอ้างอิงจากกระทู้ถามแพทย์ (สำหรับคำถามปัจจุบัน):\\n{context if context else 'ไม่พบข้อมูลอ้างอิงที่เกี่ยวข้องโดยตรงสำหรับคำถามนี้'}",
            f"\\nคำถามปัจจุบันของผู้ใช้:\n{query}"
        ]
        prompt = "\\n\\n".join(prompt_parts)

        # Run synchronous LLM call in an executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        answer_raw = await loop.run_in_executor(None, lambda: llm.invoke(prompt))

        disclaimer_text = "ข้อมูลนี้เป็นเพียงข้อมูลเบื้องต้นที่สรุปจากกระทู้ถามตอบในฟอรั่ม Agnos Health และไม่สามารถใช้แทนคำแนะนำ การวินิจฉัย หรือการรักษาจากแพทย์ผู้เชี่ยวชาญได้ หากคุณมีข้อกังวลด้านสุขภาพ กรุณาปรึกษาแพทย์โดยตรงนะคะ/ครับ"
        processed_answer = answer_raw.replace(disclaimer_text, "").strip()

        if not processed_answer or processed_answer == disclaimer_text.strip():
            if not context_parts: # If no context was found from RAG
                processed_answer = "ขออภัยค่ะ/ครับ ดิฉันไม่พบข้อมูลที่เกี่ยวข้องโดยตรงกับคำถามของคุณในฐานข้อมูล ณ ขณะนี้"
            else: # Context was found, but LLM didn't generate a meaningful answer
                processed_answer = "ข้อมูลที่มีอยู่อาจไม่เพียงพอที่จะให้คำตอบที่ชัดเจนสำหรับคำถามนี้ค่ะ/ครับ"

        final_answer_with_disclaimer = f"{processed_answer}\\n\\n{disclaimer_text}"

        # If no relevant documents were found, reflect this in the answer more clearly if not already handled
        if not results["documents"] or not results["documents"][0]:
            # Override answer if RAG found nothing, ensuring it aligns with prompt instructions for no info
            final_answer_with_disclaimer = f"ขออภัยค่ะ/ครับ ดิฉันไม่พบข้อมูลที่เกี่ยวข้องโดยตรงกับคำถามของคุณในฐานข้อมูล ณ ขณะนี้\\n\\n{disclaimer_text}"
            sources = [] # Ensure sources are empty if RAG found nothing

        return {
            "answer": final_answer_with_disclaimer,
            "sources": sources
        }
    except Exception as e:
        print(f"Error in RAG query: {e}")
        # Return a generic error message with disclaimer
        error_disclaimer = "ข้อมูลนี้เป็นเพียงข้อมูลเบื้องต้นที่สรุปจากกระทู้ถามตอบในฟอรั่ม Agnos Health และไม่สามารถใช้แทนคำแนะนำ การวินิจฉัย หรือการรักษาจากแพทย์ผู้เชี่ยวชาญได้ หากคุณมีข้อกังวลด้านสุขภาพ กรุณาปรึกษาแพทย์โดยตรงนะคะ/ครับ"
        return {
            "answer": f"เกิดข้อผิดพลาดในการประมวลผลคำถามของคุณ: {str(e)}\\n\\n{error_disclaimer}",
            "sources": []
        } 