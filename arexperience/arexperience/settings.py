from pathlib import Path
import os

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# ==============================================================================
# SECURITY SETTINGS
# ==============================================================================
SECRET_KEY = "django-insecure-hfrhzp0c872fe@&22sa^2sjw0lxjsgh86-rikmj4^npcu08h59"
DEBUG = True

ALLOWED_HOSTS = ["*"]
#  "   "127.0.0.1",
#     "localhost", 
#     ".ngrok-free.app",
#     "notifications-janet-consist-now.trycloudflare.com"
# ]

# ==============================================================================
# APPLICAT/weather-spa-herein-coupled.trycloudflare.comIONS
# ==============================================================================
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "corsheaders",  # For CORS handling
    "arexp",
    'channels',# Your AR Experience app
]

# ==============================================================================
# MIDDLEWARE
# ==============================================================================
MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.common.CommonMiddleware",
    
]

ROOT_URLCONF = "arexperience.urls"

# ==============================================================================
# TEMPLATES
# ==============================================================================
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "django.template.context_processors.media",
            ],
        },
    },
]

WSGI_APPLICATION = "arexperience.wsgi.application"

# ==============================================================================
# DATABASE
# ==============================================================================
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

# ==============================================================================
# PASSWORD VALIDATION
# ==============================================================================
AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]

# ==============================================================================
# INTERNATIONALIZATION
# ==============================================================================
LANGUAGE_CODE = "en-us"
TIME_ZONE = "Asia/Kolkata"
USE_I18N = True
USE_TZ = True

# ==============================================================================
# STATIC FILES CONFIGURATION
# ==============================================================================
STATIC_URL = '/static/'
STATICFILES_DIRS = [
    BASE_DIR / "static",
    Path("D:/git/AR-Experience-/arexperience/node_modules"),  # Full path
]
STATIC_ROOT = BASE_DIR / "staticfiles"

# ==============================================================================
# MEDIA FILES CONFIGURATION
# ==============================================================================
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Custom media directories
PREVIEWS_DIR = MEDIA_ROOT / 'previews'

# ==============================================================================
# FILE UPLOAD SETTINGS
# ==============================================================================
DATA_UPLOAD_MAX_MEMORY_SIZE = 524288000  # 500MB
FILE_UPLOAD_MAX_MEMORY_SIZE = 524288000   # 500MB

# ==============================================================================
# CORS CONFIGURATION (for MindAR)
# ==============================================================================
CORS_ALLOW_ALL_ORIGINS = True  # For development only

CORS_ALLOWED_ORIGINS = [
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "https://tamil-cute-games-range.trycloudflare.com",
]
CORS_ALLOWED_METHODS = [
    'DELETE',
    'GET',
    'OPTIONS',
    'PATCH',
    'POST',
    'PUT',
]

CORS_ALLOW_HEADERS = [
    'accept',
    'accept-encoding',
    'authorization',
    'content-type',
    'dnt',
    'origin',
    'user-agent',
    'x-csrftoken',
    'x-requested-with',
    'range',  # Important for video files
]

# ==============================================================================
# CSRF CONFIGURATION
# ==============================================================================
CSRF_TRUSTED_ORIGINS = [
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "https://folk-albuquerque-tgp-idol.trycloudflare.com",
]

CSRF_COOKIE_SECURE = False      # Set to True in production with HTTPS
CSRF_COOKIE_HTTPONLY = True
CSRF_COOKIE_SAMESITE = 'Lax'
CSRF_USE_SESSIONS = False

# ==============================================================================
# SESSION CONFIGURATION
# ==============================================================================
SESSION_COOKIE_SECURE = False   # Set to True in production with HTTPS

# ==============================================================================
# SECURITY CONFIGURATION
# ==============================================================================
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")

# ==============================================================================
# DEFAULT SETTINGS
# ==============================================================================
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels_redis.core.RedisChannelLayer',
        'CONFIG': {
            "hosts": [('127.0.0.1', 6379)],
        },
    },
}


ASGI_APPLICATION = 'yourproject.asgi.application'