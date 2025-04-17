import os
import re
import tempfile
from typing import List, Dict, Any
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import openai
import chromadb

# Updated imports for LangChain
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app) 
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize persistent ChromaDB
CHROMA_PERSIST_DIR = "./chroma_db"
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Initialize vector store with explicit client
collection_name = "nashik_police_documents"
try:
    # Create collection if it doesn't exist
    chroma_collection = chroma_client.get_or_create_collection(name=collection_name)
    
    # Initialize Chroma with the created collection
    vectorstore = Chroma(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=embeddings
    )
except Exception as e:
    print(f"Error initializing ChromaDB: {e}")
    # Fallback initialization without using existing collection
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=collection_name
    )

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Initialize language model
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.2
)

# Define Nashik Police Department specific prompt template
nashik_police_qa_template = """
You are an AI assistant for the Nashik City Police Department. Your role is to provide helpful, accurate information to the citizens of Nashik based on police documents and general knowledge about police procedures.

RULES TO FOLLOW:
1. Always be respectful, professional, and informative, representing the Nashik City Police in a positive manner.
2. For emergencies, immediately advise the user to call 100 (Police Emergency) or 112 (Universal Emergency Number) instead of using this chatbot.
3. For questions about specific criminal cases, advise users to contact the department directly through official channels.
4. Never provide legal advice - clarify that your responses are informational only.
5. If you're unsure about an answer, acknowledge the limitation rather than providing potentially incorrect information.
6. For questions about filing reports or obtaining documents, provide the general procedure as per Nashik Police guidelines but remind users that official processes must be followed.
7. When referring to contact information or locations, use the official Nashik City Police details available on their website (https://nashikcitypolice.gov.in/).
8. Maintain a helpful, official tone appropriate for a police department communication.
9. Prioritize public safety and community service in all responses.
10. If asked about traffic rules or fines, provide general information applicable in Nashik but suggest the user verify current rates and rules with the traffic department.
11. If you can't find any relevant information just whatsapp this +919923323311

SPECIAL TOPICS GUIDELINES:
- CYBER SAFETY: Provide practical tips on preventing cybercrime, securing personal data, recognizing phishing attempts, and safe internet practices. Emphasize reporting cyber crimes to the Cyber Police Station.
- CYBER AWARENESS: Educate about current cyber threats, online fraud methods, safe digital transactions, and protecting digital identity.
- POLICE RECRUITMENT: Provide general information about recruitment processes, eligibility criteria, and direct users to the official recruitment portal for current openings.
- PRESS RELEASES: Share information from recent press releases if available in the knowledge base, otherwise direct users to the official Nashik Police website's press section.
- RIGHT TO INFORMATION (RTI): Explain the basic RTI application process, fee structure, and how to submit RTI requests to the Nashik Police Department.
- MAHARASHTRA RIGHT TO SERVICE ACT: Outline citizens' rights under this act, services covered, and timeframes for service delivery as applicable to police services.
- PASSPORT STATUS: Explain the police verification process for passports, and direct users to the Passport Seva website for status checks.
- USEFUL WEBSITES: Provide links to relevant government websites like Maharashtra Police, Nashik Municipal Corporation, and emergency services.
- CITIZEN WALL: Explain what the Citizen Wall initiative is if available in the knowledge base, otherwise provide general information about citizen engagement platforms.
- TENDERS: Direct users to the official tender section of the Nashik Police website for current tenders and procurement notices.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

NASHIK_POLICE_QA_PROMPT = PromptTemplate(
    template=nashik_police_qa_template,
    input_variables=["context", "question"]
)

# Initialize retrieval QA chain with custom prompt
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": NASHIK_POLICE_QA_PROMPT}
)

def is_valid_filename(filename):
    """Check if the filename is valid and has an allowed extension."""
    allowed_extensions = {'pdf', 'txt', 'docx'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def get_loader_for_file(file_path):
    """Return the appropriate document loader based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.pdf':
        return PyPDFLoader(file_path)
    elif ext == '.txt':
        return TextLoader(file_path)
    elif ext == '.docx':
        return Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def process_document(file_path):
    """Process a document: load, split, embed, and store in the vector DB."""
    # Load document
    loader = get_loader_for_file(file_path)
    documents = loader.load()
    
    # Split document into chunks
    chunks = text_splitter.split_documents(documents)
    
    # Add chunks to the vector store
    vectorstore.add_documents(chunks)
    
    return len(chunks)

def get_fallback_response(question):
    """Provide standard responses for common topics when no relevant documents are found."""
    # Convert question to lowercase for case-insensitive matching
    question_lower = question.lower()
    
    # Check for keywords related to special topics
    if any(keyword in question_lower for keyword in ["cyber safety", "online safety", "internet safety"]):
        return """
        Cyber Safety Tips from Nashik Police:
        1. Use strong, unique passwords for all accounts
        2. Enable two-factor authentication when available
        3. Be cautious of suspicious emails, links, and attachments
        4. Keep your software and apps updated
        5. Use secure and private networks
        6. Be careful about sharing personal information online
        7. Regularly monitor your accounts for suspicious activity
        8. Report cyber crimes to Nashik Cyber Police Station
        
        For more specific guidance, please visit our official website or contact the Cyber Crime Cell at the Nashik Police Cyber Cell.
        """
    
    elif any(keyword in question_lower for keyword in ["cyber awareness", "cyber crime", "online fraud"]):
        return """
        Cyber Awareness Advisory:
        - Be alert to phishing attempts that impersonate banks, government agencies, or trusted companies
        - Verify the source before sharing OTPs or financial information
        - Use secure payment gateways for online transactions
        - Be wary of too-good-to-be-true offers and lottery/prize announcements
        - Install reputable antivirus and anti-malware software
        - Regularly check your bank statements for unauthorized transactions
        
        To report cyber crime, visit cybercrime.gov.in or contact Nashik Cyber Police.
        """
    
    elif any(keyword in question_lower for keyword in ["recruitment", "police job", "bharti", "vacancy"]):
        return """
        Police Recruitment Information:
        
        For the most current information about Nashik Police recruitment, eligibility criteria, application process, and upcoming vacancies, please visit:
        
        1. Official Nashik City Police website: https://nashikcitypolice.gov.in
        2. Maharashtra Police Recruitment portal: https://mahapolice.gov.in
        
        All official recruitment notifications are published on these websites along with detailed instructions for application.
        """
    
    elif any(keyword in question_lower for keyword in ["press", "news", "release", "media"]):
        return """
        For the latest press releases and news from Nashik City Police Department, please visit the official website at https://nashikcitypolice.gov.in and navigate to the Press Releases or Media section.
        
        Official announcements are also sometimes shared on the Nashik Police social media channels.
        """
    
    elif any(keyword in question_lower for keyword in ["rti", "right to information", "information act"]):
        return """
        Right to Information (RTI) Process:
        
        To file an RTI application with Nashik Police:
        1. Prepare an application in the prescribed format
        2. Include the required fee (₹10 for general category)
        3. Address it to the Public Information Officer (PIO), Nashik City Police
        4. Clearly specify the information you seek
        5. Submit the application in person or by post to the PIO office
        
        For more details, contact the RTI cell at Nashik Police Commissioner's office.
        """
    
    elif any(keyword in question_lower for keyword in ["right to service", "service act", "rts"]):
        return """
        Maharashtra Right To Service Act Information:
        
        This act ensures timely delivery of notified public services to citizens. For police services, this includes:
        
        1. FIR registration - Immediate
        2. Copy of FIR - Within 24 hours
        3. Police verification for passport - Within 21 working days
        4. Police clearance certificate - Within 21 working days
        5. Permission for events/processions - Within 7 working days
        
        If these services are delayed without reason, citizens can file an appeal with the designated officer.
        """
    
    elif any(keyword in question_lower for keyword in ["passport", "passport verification", "passport status"]):
        return """
        Passport Police Verification Process:
        
        1. After submitting your passport application, the Passport Seva Kendra forwards verification requests to the local police
        2. Nashik Police conducts address and identity verification
        3. You may be contacted by a police officer for physical verification
        4. Keep all original documents ready for verification
        5. For status updates, check the Passport Seva website (passportindia.gov.in)
        
        To expedite the process, ensure your address details are accurate and keep your contact information updated.
        """
    
    elif any(keyword in question_lower for keyword in ["useful websites", "websites", "official sites"]):
        return """
        Useful Websites:
        
        1. Nashik City Police: https://nashikcitypolice.gov.in
        2. Maharashtra Police: https://mahapolice.gov.in
        3. Nashik Municipal Corporation: https://nashikcorporation.gov.in
        4. Cyber Crime Reporting: https://cybercrime.gov.in
        5. National Crime Records Bureau: https://ncrb.gov.in
        6. Emergency Services: https://108.life (Ambulance), https://nashikfirebrigade.com (Fire)
        7. e-Courts: https://districts.ecourts.gov.in/nashik
        8. Traffic Management: https://mahatrafficpolice.gov.in
        """
    
    elif any(keyword in question_lower for keyword in ["citizen wall", "citizen portal", "citizen feedback"]):
        return """
        Citizen Wall Initiative:
        
        The Citizen Wall is a platform for public engagement with the Nashik Police Department. Through this initiative, citizens can:
        
        1. Share feedback about police services
        2. Suggest improvements to community safety
        3. Participate in community policing programs
        4. Access information about police-community partnerships
        
        To participate, visit the Citizen Section on the official Nashik Police website or follow the department's social media channels.
        """
    
    elif any(keyword in question_lower for keyword in ["tender", "bid", "contract", "procurement"]):
        return """
        Information About Tenders:
        
        All current tenders, RFPs, and procurement notices from Nashik City Police are published on:
        
        1. The official Nashik City Police website: https://nashikcitypolice.gov.in/tenders
        2. The Maharashtra Government Tender Portal: https://mahatenders.gov.in
        
        For specific tender details including eligibility, requirements, submission deadlines, and contact information, please refer to the official tender documents published on these platforms.
        """
    
    # Return None if no matching topic is found
    return None

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle document upload and processing."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not is_valid_filename(file.filename):
        return jsonify({"error": "Invalid file type. Supported types: PDF, TXT, DOCX"}), 400
    
    # Save the file temporarily
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    
    try:
        # Process the document
        chunk_count = process_document(file_path)
        
        # Clean up
        os.remove(file_path)
        
        return jsonify({
            "message": f"Document processed successfully: {file.filename}",
            "chunks": chunk_count
        })
    except Exception as e:
        # Clean up on error
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({"error": str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    """Handle Q&A based on the uploaded documents."""
    data = request.json
    
    if not data or 'question' not in data:
        return jsonify({"error": "No question provided"}), 400
    
    question = data['question']
    
    # Check for emergency keywords in multiple languages (English, Hindi, Marathi)
    emergency_keywords = [
        "emergency", "help me", "urgent", "dying", "danger", "immediate assistance", 
        "आपातकालीन", "मदद करो", "तत्काल", "खतरा", "तात्काल सहायता",
        "आणीबाणी", "मदत करा", "तातडीचे", "धोका", "तात्काळ मदत"
    ]
    
    if any(keyword in question.lower() for keyword in emergency_keywords):
        return jsonify({
            "answer": "महत्वपूर्ण सूचना: हि आणीबाणीची परिस्थिती असल्यास, कृपया तात्काळ 100 (पोलीस) किंवा 112 (सार्वत्रिक आपात्कालीन क्रमांक) वर फोन करा. हा चॅटबॉट आपात्कालीन प्रतिसादासाठी नियंत्रित केलेला नाही.\n\nIMPORTANT: If this is an emergency, please call 100 (Police) or 112 (Universal Emergency Number) immediately. This chatbot is not monitored for emergency response.",
            "sources": []
        })
    
    try:
        # Execute the retrieval QA chain
        result = qa_chain.invoke({"query": question})
        
        # Extract the source documents
        source_docs = result.get("source_documents", [])
        sources = []
        
        for doc in source_docs:
            source_info = {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            sources.append(source_info)
        
        # Check if the answer is too generic or doesn't have enough information
        answer = result["result"]
        if ("I don't have enough information" in answer or 
            "I don't have specific information" in answer or
            "cannot find relevant information" in answer or
            "whatsapp +919923323311" in answer):
            
            # Try to get a fallback response for common topics
            fallback = get_fallback_response(question)
            if fallback:
                answer = fallback
        
        return jsonify({
            "answer": answer,
            "sources": sources
        })
    except Exception as e:
        # Check if we can provide a fallback response even when the chain fails
        fallback = get_fallback_response(question)
        if fallback:
            return jsonify({
                "answer": fallback,
                "sources": []
            })
        return jsonify({"error": str(e)}), 500

@app.route('/documents', methods=['GET'])
def list_documents():
    """List all documents in the vector store."""
    try:
        # Retrieve all documents from the vector store
        docs = vectorstore.get()
        
        # Extract unique document sources from metadata
        unique_docs = set()
        for i, metadata in enumerate(docs.get("metadatas", [])):
            if "source" in metadata:
                unique_docs.add(metadata["source"])
        
        return jsonify({"documents": list(unique_docs)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/documents/<path:doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    """Delete a document from the vector store."""
    try:
        # Delete the document
        vectorstore.delete(filter={"source": doc_id})
        vectorstore.persist()
        
        return jsonify({"message": f"Document deleted: {doc_id}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index2.html')

# Add a welcome message endpoint for chatbot initialization
@app.route('/welcome', methods=['GET'])
def welcome_message():
    """Return a welcome message when the chatbot is first loaded."""
    return jsonify({
        "message": "नमस्कार! नाशिक शहर पोलिस विभागाच्या सहाय्यक चॅटबॉटमध्ये आपले स्वागत आहे. आपण पोलिस सेवा, तक्रारी नोंदवणे, किंवा माहिती विचारू शकता. आपात्कालीन परिस्थितीत कृपया 100 किंवा 112 वर कॉल करा.\n\nWelcome to the Nashik City Police Department Assistant. You can ask questions about police services, report non-emergency issues, or request information. For emergencies, please call 100 or 112.",
        "suggestions": [
            "पोलीस तक्रार कशी नोंदवावी? / How do I file a police report?",
            "नाशिक पोलीस स्टेशनचे कार्यालयीन वेळ / Nashik police station timings",
            "सायबर सुरक्षा टिप्स / Cyber safety tips",
            "पोलीस भरती माहिती / Police recruitment information",
            "आरटीआय अर्ज कसा करावा? / How to file an RTI application?",
            "निविदा प्रक्रिया / Tender process"
        ]
    })

# Add a new endpoint for checking system health
@app.route('/health', methods=['GET'])
def health_check():
    """Return system health status."""
    try:
        # Check if ChromaDB is accessible
        chroma_status = "OK" if chroma_client.heartbeat() else "Error"
        
        # Check if vector store has documents
        doc_count = len(vectorstore.get().get("ids", []))
        
        return jsonify({
            "status": "healthy",
            "chroma_db": chroma_status,
            "document_count": doc_count,
            "api_version": "1.0.0"
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

# Add an endpoint for topic suggestions based on user input
@app.route('/suggest', methods=['POST'])
def suggest_topics():
    """Suggest relevant topics based on user input."""
    data = request.json
    
    if not data or 'input' not in data:
        return jsonify({"error": "No input provided"}), 400
    
    user_input = data['input'].lower()
    
    # Define topic categories and related keywords
    topic_suggestions = {
        "cyber": ["cyber safety tips", "how to report cyber crime", "online fraud prevention"],
        "recruitment": ["police recruitment process", "eligibility criteria", "upcoming vacancies"],
        "complaints": ["how to file a complaint", "track complaint status", "contact nearest police station"],
        "documents": ["document verification process", "required documents for police certificate"],
        "traffic": ["traffic rules in Nashik", "pay traffic fines online", "vehicle towing information"],
        "rti": ["RTI application process", "fees for RTI", "appeal process"],
        "passport": ["police verification for passport", "check passport verification status"],
        "services": ["police clearance certificate", "event permission application", "tenant verification"]
    }
    
    # Match user input with topics
    suggestions = []
    for category, topics in topic_suggestions.items():
        if any(keyword in user_input for keyword in [category, *category.split()]):
            suggestions.extend(topics)
    
    # If no specific matches, return general suggestions
    if not suggestions:
        suggestions = [
            "file a police complaint",
            "cyber safety tips",
            "police station contact information",
            "police verification services",
            "important contact numbers"
        ]
    
    # Limit to 5 suggestions
    suggestions = suggestions[:5]
    
    return jsonify({"suggestions": suggestions})

if __name__ == '__main__':
    app.run(debug=True)