# Quandl uses a very straightforward API to make its free and premium data available. Currently, 50 anonymous calls are
# allowed, then a (free) API key is required. See [documentation](https://www.quandl.com/tools/api) for more details.
import os
import quandl
import seaborn as sns
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv(verbose=True)
sns.set_style("whitegrid")

api_key = os.getenv("quandl")
oil = quandl.get("EIA/PET_RWTC_D", api_key=api_key).squeeze()

oil.plot(lw=2, title="WTI Crude Oil Price", figsize=(12, 4))
plt.tight_layout()
plt.savefig("images/04-01.png", bboxinches="tight")
