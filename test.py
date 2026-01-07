import os 
import sys 
from backend.src.document_chat.retrieval import ConversationalRAG
from backend.utils.model_loader import ModelLoader
from backend.src.document_ingestion.data_ingestion import ChatIngestor
from dotenv import load_dotenv
from pathlib import Path
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()



def test_doc_ingestion_and_rag():
    try:
        test_file='/Users/sachinbeast/Desktop/main/system/VMP_final/data/xx.pdf'
        uploaded_file=[]

        for filepath in test_file:
            if Path(filepath).is_file():
                f=open(filepath,'rb')
                uploaded_file.append(f)
                print('done')
            else:
                print(f"File not found: {filepath}")
        print('--------------')
        
        if not uploaded_file:
            print("No valid files found for ingestion")
            sys.exit(1)

        print('File Uplaoded successfully')

        ci=ChatIngestor(temp_base='data',faiss_base='faiss_index',use_session_dirs=True)

        ret=ci.built_retriver(uploaded_files=uploaded_file,
        chunk_size=100,
        chunk_overlap=20,
        k=5,
        search_type='mmr',
        fetch_k=20,
        lambda_mult=0.5
        )


        for f in uploaded_file:
            try:
                f.close()
            except Exception as e:
                print(f"Failed to close file: {f}")
        
        session_id=ci.session_id

        index_dir=os.path.join('faiss_index',session_id)

        rag=ConversationalRAG(session_id=session_id,retriever=ret)

        # rag.load_retriever_from_faiss(
        #     index_path=index_dir, 
        #     k=5, 
        #     index_name=os.getenv("FAISS_INDEX_NAME", "index"),
        #     search_type="mmr",
        #     fetch_k=20,
        #     lambda_mult=0.5
        # )

        chat_history=[]

        print('type exit to quit')

        while True:
            try:
                user_input=input('You: ').strip()
                if user_input.lower()=='exit':
                    break
                
            except (EOFError, KeyboardInterrupt):
                print("Bye!")
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "q", ":q"}:
                print("Goodbye!")
                break
            
            answer=rag.invoke(user_input,chat_history=chat_history)
            print('Assistant: ',answer)

            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=answer))
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__=='__main__':
    test_doc_ingestion_and_rag()






        
        

        