# CrescendoAttacker
Based on the [paper](https://arxiv.org/abs/2404.01833) and [research](https://crescendo-the-multiturn-jailbreak.github.io//) performed by Mark Russinovich and his team, this is a very rudimentary implementation that allows users to generate prompt sets from a library. That prompt set can then be used by crescendoattacker.py to test against models specified in the code. The code is currently set to target OpenAI's GPT 3.5 and requires an OpenAI API key to be set as an environment variable OPENAI_API_KEY.

# How-To
1. Run PromptGenerator.py and wait for the crescendo_prompts.json file to be generated
2. Configure CrescendoAttacker.py to target your model of choice and make sure your API key is populated in the environment variable
3. Run CrescendoAttacker.py and observe output

# Inspiration
Inspiration for this project came from wanting to automate model evaluation frameworks in an effort to expedite assessments and susceptibility of models to crescendo attacks. This is a very simplistic approach and I would like to build upon this in the future by adding:
- Error Handling
- Persistent storage for testing results
- Easier adapatability for non-OpenAI model testing
- Fine-tuning opening -> escalation -> intensifier process

Eventually I would like to train an adversarial model that is able to perform this testing and interpret how much or how little to escalate based on the target model's responses.
