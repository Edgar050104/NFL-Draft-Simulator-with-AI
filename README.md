NFL Draft Simulator with AI
An advanced NFL draft simulator that uses Bayesian networks, fuzzy logic, and machine learning elements to make intelligent and realistic draft decisions.

📋 Description

This project implements a comprehensive NFL draft simulator that mimics the decision-making process of real teams. Using advanced artificial intelligence techniques, the system evaluates prospects based on:

- Player rankings (prospect quality)
- Team-specific needs
- Historical draft selection patterns by position and round

The combination of Bayesian networks to model historical probabilities, fuzzy logic to simulate human reasoning, and machine learning elements for optimization creates a system capable of making realistic selections and providing reasoning behind each decision.

✨ Features

Bayesian Network: Learns from historical data to calculate selection probabilities by position and round
Fuzzy Logic System: Implements linguistic rules like "if need is high and ranking is elite, then fit is high"
Interactive Dashboard: Streamlit graphical interface to visualize and analyze results
Customization: Allows modification of team needs and factor weights
Advanced Visualizations: Charts, diagrams, and heat maps for detailed analysis
Complete Simulation: Ability to simulate all 257+ picks of the draft

🚀 Installation
bash# Clone the repository
git clone https://github.com/username/nfl-draft-simulator.git
cd nfl-draft-simulator

# Install dependencies
pip install -r requirements.txt

📊 Usage
Console Simulator
For a quick demonstration through the console:
bashpython example_simulator.py
This script showcases the basic capabilities of the simulator, including probability calculation, fit scores, and pick simulation.
Interactive Dashboard
For the complete experience with graphical interface:
bashstreamlit run dashboard.py
The dashboard allows you to:

Configure the number of picks to simulate
Customize decision factor weightings
Modify team needs
Visualize results with interactive charts
Analyze data by team, position, and round

📁 Project Structure

nfl_bayesian_fuzzy.py: Main simulator implementation (classes for Bayesian Network, Fuzzy System, and Simulator)
example_simulator.py: Console demonstration script
dashboard.py: Streamlit application for the graphical interface
requirements.txt: Project dependencies
CSV data files:

prospects.csv: List of prospects with their rankings
team_needs.csv: Needs of each team by position
trends.csv: Historical draft trends
draft_order.csv: Complete draft order

📊 Results
The simulator has demonstrated:

High correlation (0.83) between simulated positional distribution and historical patterns
76% satisfaction of priority needs in the first 3 rounds
Average fit score of 72.5/100 for all selections

🛠️ Requirements

Python 3.10 or higher (3.10 or 3.11 recommended for best compatibility)
Dependencies listed in requirements.txt:

scikit-fuzzy
matplotlib
networkx
numpy
pandas
scikit-learn
seaborn
streamlit

🤝 Contributing
Contributions are welcome. For major changes, please open an issue first to discuss what you would like to change.

Fork the repository
Create your feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add some amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request
