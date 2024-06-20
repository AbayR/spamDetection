from fastapi import FastAPI, Depends, HTTPException, status, Header, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy import Column, Integer, String, ForeignKey, Boolean, select
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
import joblib
from sqlalchemy import text
import logging
import string
from bs4 import BeautifulSoup
from fastapi.templating import Jinja2Templates

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load the model and vectorizer
pipeline = joblib.load('spam_detector_pipeline.pkl')

# Database setup
DATABASE_URL = "sqlite+aiosqlite:///./spam.db"
Base = declarative_base()

SECRET_KEY = "your-secure-randomly-generated-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

engine = create_async_engine(DATABASE_URL, echo=True)
async_session_maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

templates = Jinja2Templates(directory="templates")


# Dependency to get DB session
async def get_db():
    async with async_session_maker() as session:
        yield session


# User models and schemas
class Token(BaseModel):
    access_token: str
    token_type: str


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    spam_rules = relationship("SpamRule", back_populates="user")
    predictions = relationship("Prediction", back_populates="user")


class SpamRule(Base):
    __tablename__ = "spam_rules"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # user_id is nullable
    rule = Column(String, index=True)
    user = relationship("User", back_populates="spam_rules")


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    message = Column(String)
    prediction = Column(String)
    user = relationship("User", back_populates="predictions")


class UserCreate(BaseModel):
    email: EmailStr
    password: str


class Message(BaseModel):
    text: str


class CustomRule(BaseModel):
    rule: str


# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust the allowed origins as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Predefined list of common English stopwords
common_stopwords = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
                    'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
                    'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
                    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
                    'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
                    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
                    'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
                    'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
                    'should', 'now'}


# Text preprocessing function
def preprocess_text_simple(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize and remove stopwords
    tokens = [word for word in text.split() if word not in common_stopwords]
    return ' '.join(tokens)


# Validation function to clean the input text
def clean_text(text):
    # Remove unwanted characters and extra spaces
    text = text.replace("'", "").replace('"', "").replace("/", "").replace("  ", " ")
    return text


# Function to extract text from HTML
def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.get_text(separator=' ')


# Password handling functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def create_access_token(data: dict):
    to_encode = data.copy()
    # Remove or comment out the expiration setting
    # expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    # to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_user(db: AsyncSession, email: str):
    result = await db.execute(select(User).filter(User.email == email))
    return result.scalars().first()


async def authenticate_user(db: AsyncSession, email: str, password: str):
    user = await get_user(db, email)
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user


async def get_current_user(authorization: str = Header(None), db: AsyncSession = Depends(get_db)):
    logger.debug(f"Authorization header: {authorization}")
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if authorization is None or not authorization.startswith("Bearer "):
        logger.debug("Authorization header is missing or does not start with 'Bearer '")
        raise credentials_exception
    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        logger.debug(f"Decoded JWT payload: {payload}")
        if email is None:
            logger.debug("Email is None in JWT payload")
            raise credentials_exception
    except JWTError as e:
        logger.debug(f"JWT decode error: {e}")
        raise credentials_exception
    user = await get_user(db, email)
    if user is None:
        logger.debug("User not found")
        raise credentials_exception
    logger.debug(f"Authenticated user: {user.email}")
    return user


# Routes
@app.post('/token', response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    user = await authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.email})
    logger.debug(f"Generated access token: {access_token}")
    return {"access_token": access_token, "token_type": "bearer"}


@app.post('/register', response_model=dict)
async def create_user(user: UserCreate, db: AsyncSession = Depends(get_db)):
    user_db = await get_user(db, user.email)
    if user_db:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = get_password_hash(user.password)
    db_user = User(email=user.email, hashed_password=hashed_password)
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    logger.debug(f"User created successfully: {user.email}")
    return {"msg": "User created successfully"}


@app.post('/predict_public/')
async def predict_spam_public(message: Message):
    logger.debug(f"Received public prediction request with message: {message.text}")
    cleaned_text = clean_text(message.text)
    extracted_text = extract_text_from_html(cleaned_text)
    preprocessed_text = preprocess_text_simple(extracted_text)
    text_vectorized = pipeline.named_steps['vectorizer'].transform([preprocessed_text])
    prediction = pipeline.named_steps['classifier'].predict(text_vectorized)[0]
    logger.debug(f"Prediction result: {prediction}")
    return {"prediction": prediction}


@app.post('/predict/')
async def predict_spam(message: Message, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    logger.debug(f"Received prediction request with message: {message.text} from user: {user.email}")
    cleaned_text = clean_text(message.text)
    extracted_text = extract_text_from_html(cleaned_text)
    preprocessed_text = preprocess_text_simple(extracted_text)

    custom_rules = await db.execute(select(SpamRule.rule).where(SpamRule.user_id == user.id))
    custom_rules = custom_rules.scalars().all()
    for rule in custom_rules:
        if rule in preprocessed_text:
            prediction = "spam"
            break
    else:
        text_vectorized = pipeline.named_steps['vectorizer'].transform([preprocessed_text])
        prediction = pipeline.named_steps['classifier'].predict(text_vectorized)[0]

    new_prediction = Prediction(user_id=user.id, message=message.text, prediction=prediction)
    db.add(new_prediction)
    await db.commit()

    logger.debug(f"Prediction result: {prediction}")
    return {"prediction": prediction}


@app.post('/custom_rules/')
async def add_custom_rule(custom_rule: CustomRule, db: AsyncSession = Depends(get_db)):
    logger.debug(f"Received custom rule addition request: {custom_rule.rule}")
    new_rule = SpamRule(user_id=None, rule=custom_rule.rule)  # user_id is set to None as it is no longer required
    db.add(new_rule)
    await db.commit()
    return {"status": "rule added"}


@app.get('/users/', response_class=HTMLResponse)
async def get_users(request: Request, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User))
    users = result.scalars().all()
    return templates.TemplateResponse("users.html", {"request": request, "users": users})


@app.get('/custom_rules/', response_class=HTMLResponse)
async def get_custom_rules(request: Request, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(SpamRule))
    rules = result.scalars().all()
    return templates.TemplateResponse("custom_rules.html", {"request": request, "rules": rules})


@app.get('/report/', response_class=HTMLResponse)
async def get_report(request: Request, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Prediction))
    predictions = result.scalars().all()
    return templates.TemplateResponse("report.html", {"request": request, "predictions": predictions})


@app.delete("/delete_all/")
async def delete_all_values(db: AsyncSession = Depends(get_db)):
    try:
        await db.execute(text('DELETE FROM predictions'))
        await db.execute(text('DELETE FROM spam_rules'))
        await db.execute(text('DELETE FROM users'))
        await db.commit()
        return {"status": "All values deleted successfully"}
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


# Create the database tables
@app.on_event("startup")
async def on_startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
