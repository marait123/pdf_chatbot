import environ
from functools import lru_cache
from dotenv import load_dotenv
load_dotenv()
ENV_CONFIG_PREFIX = 'APP'


@environ.config(frozen=True)
class ModelConfig:
    model_name = environ.var(default="gemini-1.5-flash")
    temperature = environ.var(default=0.0)
    max_tokens = environ.var(default=None)
    timeout = environ.var(default=None)
    max_retries = environ.var(default=2)


@environ.config(frozen=True)
class TrimmerConfig:
    max_tokens = environ.var(default=100000)
    strategy = environ.var(default="last")
    include_system = environ.var(default=True)
    allow_partial = environ.var(default=False)
    start_on = environ.var(default="human")


@environ.config(frozen=True)
class MlConfig:
    bucket_name = environ.var(default="genai")
    model = environ.group(ModelConfig)
    trimmer = environ.group(TrimmerConfig)
    chunk_size = environ.var(default=18*1000)
    chunk_overlap = environ.var(default=2000)


@environ.config(prefix=ENV_CONFIG_PREFIX, frozen=True)
class Config:
    LOG_LEVEL = environ.var(default='INFO')
    ML: MlConfig = environ.group(MlConfig)
    GOOGLE_API_KEY = environ.var()
    MAX_CONTEXT_WINDOW = environ.var(default=512)
    FAISS_DB_PATH = environ.var(default="./faiss_index")
    UPLOADS_DIR = environ.var(default="./uploads")
    STATIC_DIR = environ.var(default="./static")
    TEMPLATES_DIR = environ.var(default="./templates")
    FRONTEND_INDEX = environ.var(default="./frontend/index.html")


@lru_cache(1)
def get_config():
    import os
    os.environ['GOOGLE_API_KEY'] = os.environ.get('APP_GOOGLE_API_KEY')
    conf = environ.to_config(Config)
    return conf


def show_config_help():
    import environ
    print(environ.generate_help(Config, display_defaults=True))  # noqa


# NOTE: uncomment to see the help message
# and run the script `python config.py`
if __name__ == '__main__':
    show_config_help()
