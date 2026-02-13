# UCLA Galactic Center Group StarMap

The GCG-StarMap dashboard is a dash/plotly app used to view star and orbit data from UCLA's Galactic Center Group.

## Installation

![GCG-StarMap Dashboard](assets/dashboard_example.png)

Requires Python >= 3.13.7. The -e flag installs the package in "editable" mode, so changes in the source code are immediately reflected without reinstalling.

#### MacOS / Linux

```bash
# Clone repository into desired directory
$ git clone https://github.com/brmalkan/ghez-dashboard.git .

# Create virtual environment
$ python -m venv .venv

# Activate virtual environment
$ source ~/.venv/bin/activate

# Install dependencies
$ pip install -e .
```

#### Windows

```powershell
# Clone repository into desired directory
git clone https://github.com/brmalkan/ghez-dashboard.git .

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Usage

#### MacOS / Linux

```bash
# Confirm virtual environment is active; if not, activate it
$ source ~/.venv/bin/activate

# Run the app
$ python3 src/main.py
```

#### Windows

```powershell
# Confirm virtual environment is active; if not, activate it
.\.venv\Scripts\activate

# Run the app
python src\main.py
```

App runs on port 8050 by default; viewable via browser on localhost.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
