# ─────────────────────────────────────────────
#  config.py  —  Change your settings here
# ─────────────────────────────────────────────

# Ollama model to use.
# Run `ollama list` to see what you have pulled.
#
# Options (pull with: ollama pull <model>):
#   "llama3.1:8b"    — recommended, needs ~8GB RAM
#   "llama3.1:70b"   — best quality, needs ~40GB RAM
#   "mistral:7b"     — fast, needs ~5GB RAM
#   "gemma2:9b"      — good alternative
#   "phi3:mini"      — very fast, lower quality
MODEL_NAME = "llama3.1:8b"

# Ollama server URL (default when running locally)
OLLAMA_BASE_URL = "http://localhost:11434"

# Number of debate rounds between Advocate and Critic
# 1 = fast, 2 = balanced, 3 = deep dive
DEFAULT_ROUNDS = 2

# How many web searches the Researcher runs
SEARCH_QUERIES_COUNT = 5

# Max tokens per agent response
MAX_TOKENS = 1500

# Temperature settings per agent
RESEARCHER_TEMP = 0.1   # factual — low temp
ADVOCATE_TEMP   = 0.7   # persuasive — medium
CRITIC_TEMP     = 0.6   # analytical — medium
JUDGE_TEMP      = 0.3   # structured — low