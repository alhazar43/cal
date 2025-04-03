# Short Paper Strategy  – RecSys 2025
### Positioning Strategy
The paper's goal is to introduce a conceptual and technical bridge between two domains:

1. Psychometric trait estimation (using IRT & adaptive testing)

2. Sequential, RL-based recommendation

While RL-based RecSys papers are no longer a trending topic, I can anchor the novelty of my work in the following:

Unique data source:
Most RL-based recommenders learn from behavioral feedback (clicks, views). I am introducing psychometric trait estimates, dynamically constructed from questionnaire responses, as part of the recommendation state. This is rarely explored in RecSys.

Sequential personalization:
The user's latent profile ($\theta_t$) is not static — it is incrementally learned via adaptive testing and used immediately to drive recommendation decisions.
The key is:
➡️ The RL policy learns how to recommend under uncertainty about user traits early on
➡️ And how to adapt recommendations as the trait estimate improves

Interactive learning loop:
Unlike standard psychometric studies where questionnaires are evaluated offline and applied afterwards, my system is interactive:
The questionnaire and recommendation system co-evolve in real-time.

What I want to highlight in the paper
Novel combination of adaptive testing and RL:
IRT-based trait estimation and adaptive testing have been well-studied in psychometrics, but almost never integrated into an online recommendation loop.

Cold-start personalization without history:
The system starts without any user interaction history. Instead, it rapidly builds a personalized user profile through a few targeted questions.

Proof-of-concept simulation:
I don’t need to claim real-world impact yet. The contribution is conceptual + technical — demonstrating that this two-way interaction between psychometric estimation and RL policy learning can work.

New way to model user state:
In RL RecSys literature, the state is often based on past item interactions. Here, the state is constructed from latent trait estimates updated dynamically, bringing in psychometric modeling as part of the RL loop.

How I will structure this in the paper
Introduction:
Focus on the cold-start personalization challenge, and highlight how psychometric data is an underused source in RecSys. Pitch the paper as an early-stage, novel framework rather than a production-ready system.

Method:
Keep the technical part simple:

IRT-based adaptive testing produces $\theta_t$

RL agent uses $\theta_t$ to recommend

User feedback is simulated, but the interaction loop is clear

Experiments:
Position the simulation as a controlled environment to test the concept.
Evaluation is about:

How fast the trait estimate converges

How recommendation success improves over time

How adaptive questioning beats random questioning

Conclusion:
Emphasize that this is a conceptual contribution with potential for:

Real-world deployment in high-stakes domains (like job recommendation)

More human-centric personalization models

Inspiring further research on integrating psychometric models with recommender systems