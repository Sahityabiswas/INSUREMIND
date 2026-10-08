# Insurance Sales RL Agent: Detailed Presentation Notes

**Presentation:** Interim Project Update  
**Presenter:** Sahitya Biswas  
**Registration No.:** B2530086  
**Course:** MSc in Data Science and Artificial Intelligence  
**University:** Ramakrishna Mission Vivekananda Educational and Research Institute  
**Guide:** Champak Dutta  
**Prepared:** 8 October 2026

These notes explain the current implementation and the ten slides in [the presentation](<D:/insurence seller/insurance_sales_agent/presentations/Insurance_Sales_RL_Interim_Presentation.pptx>). Each slide has a short speaking script followed by supporting material for questions. Use the scripts for the ten-minute presentation and the detailed sections for preparation. Reading the whole document aloud would take substantially longer.

The central project idea is: **understand the buyer, select a useful conversational action with PPO, filter product information through rules, and generate a response consistent with that action.** The current evidence comes from synthetic data, simulated customers, and a local text demonstration.

## Contents

- [Presentation timing](#presentation-timing)
- [Slide 1: Introduction](#slide-1-introduction)
- [Slide 2: Problem statement and objectives](#slide-2-problem-statement-and-objectives)
- [Slide 3: Literature survey](#slide-3-literature-survey)
- [Slide 4: Implemented approach](#slide-4-implemented-approach)
- [Slide 5: Dataset and customer understanding](#slide-5-dataset-and-customer-understanding)
- [Slide 6: System pipeline](#slide-6-system-pipeline)
- [Slide 7: Reinforcement learning formulation](#slide-7-reinforcement-learning-formulation)
- [Slide 8: PPO implementation in detail](#slide-8-ppo-implementation-in-detail)
- [Slide 9: Evaluation results and interpretation](#slide-9-evaluation-results-and-interpretation)
- [Slide 10: Demonstration and remaining work](#slide-10-demonstration-and-remaining-work)
- [Questions faculty may ask](#questions-faculty-may-ask)
- [Quick reference before presenting](#quick-reference-before-presenting)
- [Terminology](#terminology)
- [Source map and references](#source-map-and-references)

## Presentation Timing

| Slide | Topic | Speaking time | Cumulative time |
| --- | --- | --- | --- |
| 1 | Introduction | 30 seconds | 0:30 |
| 2 | Problem statement and objectives | 50 seconds | 1:20 |
| 3 | Literature survey | 70 seconds | 2:30 |
| 4 | Implemented approach | 50 seconds | 3:20 |
| 5 | Dataset and customer understanding | 50 seconds | 4:10 |
| 6 | System pipeline | 60 seconds | 5:10 |
| 7 | RL formulation | 75 seconds | 6:25 |
| 8 | PPO implementation | 85 seconds | 7:50 |
| 9 | Evaluation results | 70 seconds | 9:00 |
| 10 | Demonstration and remaining work | 60 seconds | 10:00 |

The timings are rehearsal targets. Pause at the pipeline and results charts, and move detailed equations or implementation questions into the discussion if time is short.

## Slide 1: Introduction

### Speaking Script

> Good morning. I am Sahitya Biswas, registration number B2530086, from MSc in Data Science and Artificial Intelligence. My project, under the guidance of Champak Dutta, is an Insurance Sales Reinforcement Learning Agent. This interim presentation covers the problem, supporting research, implemented workflow and initial evaluation. The system uses PPO to choose a conversational sales action, a product engine to control eligible information, and a local language model to express the response.

### What the Project Is Building

The project is a conversational research prototype for discussing insurance needs with a buyer. A buyer might ask about family health cover, say that a plan is expensive, mention existing insurance, or decline to continue. The system maintains context and selects a suitable next action.

The main research component is a **policy for selecting conversational strategies**. Examples include asking a clarifying question, discovering a coverage gap, handling an objection, explaining an eligible product, and respecting rejection.

The current prototype supports text interaction. Its product catalogue is synthetic. A simulated purchase in the experiments is an environment outcome, while a real sale would require an insurer's approved products, pricing and transaction process.

### Transition

> The first question is why insurance conversations need a strategy-selection component.

## Slide 2: Problem Statement and Objectives

### Speaking Script

> Insurance conversations involve a sequence of decisions. A customer may need information, reassurance, clarification, or time to consider an option. Their next useful response depends on their need, budget, existing cover and objections. My research question is whether a PPO policy can use this context to choose better sales actions in a simulated conversation. The objective includes engagement, objection resolution and suitable outcomes. It also penalizes repeated responses, pressure and unsupported information. I compare the learned policy with random, rule-based and supervised policies to assess what reinforcement learning contributes.

### Full Problem Statement

An insurance conversation cannot be evaluated only by whether each response sounds fluent. The agent also needs to choose an appropriate purpose for that response. Recommending a product before understanding the need can be unhelpful. Asking for commitment while a buyer is confused can create pressure. Repeating a product description when the buyer asks about payment fails to answer the current question.

The project therefore treats the dialogue as a sequence of decisions. At each turn, the agent observes a structured representation of the buyer and conversation, selects an action, and receives feedback during simulated training. The learning objective considers the longer-term consequences of the action sequence.

The practical research problem is to learn an adaptive policy while keeping product eligibility, factual information and explicit stop requests under separate controls.

### Research Question

> Can an intent- and emotion-aware PPO policy improve insurance sales strategy selection in a simulated multi-turn environment while maintaining suitability and low-pressure behavior?

This is a question being investigated. The current results provide preliminary evidence and do not establish that every part of the hypothesis holds.

### Objectives and How They Are Evaluated

| Objective | Implementation | Evidence available |
| --- | --- | --- |
| Understand the buyer | Intent, emotion, objection and stage classifiers, plus entity extraction | Held-out synthetic NLP metrics |
| Maintain conversation context | Structured state and compact memory | State encoding and live conversation trace |
| Select actions adaptively | Supervised initialization followed by PPO | Trained checkpoints and comparison with baselines |
| Restrict unsuitable recommendations | Need, age and budget filtering plus action masks | Product eligibility checks and simulator counters |
| Control response content | Strategy-constrained generation and response checks | Templates, local LLM demonstration and rejection diagnostics |
| Examine which components matter | Independently retrained ablations | Ablation metrics over three seeds |

### A Simple Example

Buyer: "I already have health insurance."

Possible useful action: ask about existing coverage and the reason for considering additional protection.

Possible unhelpful action: immediately ask the buyer to purchase another policy.

The research problem is to learn when a particular action is useful. The implementation also applies explicit constraints to actions that should be unavailable.

### Scope of the Contribution

The contribution is an implemented experimental framework: an insurance strategy taxonomy, a synthetic annotated dataset, a structured state representation, a customer simulator, a PPO policy, controlled generation, and reproducible evaluation. PPO itself is an existing algorithm. The project does not claim to invent a new reinforcement learning algorithm or establish a first-ever insurance negotiation system.

### Transition

> The literature provides several ways to model adaptive negotiation and language-based persuasion. I selected five studies that connect closely to this implementation.

## Slide 3: Literature Survey

### Speaking Script

> The survey follows my AI Negotiation Research Insurance document and its corrected research notes. Bagga and colleagues use actor-critic learning with supervised pre-training, which relates to my policy initialization. Sengupta and colleagues adapt strategies to opponent behavior, motivating context-dependent action selection. Renting and colleagues use graph-based policies to transfer across negotiation problems, highlighting a limitation of my fixed-size policy. Karande and colleagues study LLM persuasion with supporting agents, including fact validation and insurance examples. AgenticPay provides a benchmark for buyer-seller negotiation. Together, these studies motivate the architecture and evaluation choices. My current implementation focuses on controlled sales dialogue actions, and its results come from its own simulator.

### Source and Selection

The supplied review is [AI_Negotiation_Research_Insurance_v2.docx](<D:/insurence seller/AI_Negotiation_Research_Insurance_v2.docx>). It contains fifteen papers and a strategy reference. The presentation selects five relevant studies, using the **Verified Research Notes** column where it corrects the original summaries.

Several insurance claim-settlement examples in that document illustrate possible applications of general negotiation research. They are not automatically case studies conducted by the paper authors. This project concerns sales conversations and does not negotiate claim payouts.

### Study 1: Concurrent Bilateral Negotiation

**Paper:** Pallavi Bagga, Nicola Paoletti, Bedour Alrayes and Kostas Stathis, *A Deep Reinforcement Learning Approach to Concurrent Bilateral Negotiation*, IJCAI 2020, pp. 297-303. [Primary source](https://www.ijcai.org/proceedings/2020/42).

The paper applies an actor-critic architecture to concurrent bilateral negotiation in changing e-markets. It uses supervised pre-training on synthetic market data to reduce the initial exploration needed for learning.

**Connection:** The project also initializes its policy from supervised action examples before reinforcement learning.

**Difference:** The published setting involves concurrent market negotiations and a deep policy. This implementation uses a linear policy for a single insurance conversation.

**Correction to remember:** The paper's method is actor-critic. The original review column's DQN description is incorrect.

**Useful oral explanation:** "This study supports combining an initial supervised policy with subsequent learning from negotiation outcomes."

### Study 2: Adaptive Strategy Switching

**Paper:** Ayan Sengupta, Yasser Mohammad and Shinji Nakadai, *An Autonomous Negotiating Agent Framework with Reinforcement Learning Based Strategies and Adaptive Strategy Switching Mechanism*, AAMAS 2021. [Primary source](https://arxiv.org/abs/2102.03588).

The framework classifies opponent behavior during a negotiation and selects, switches or combines strategies. Its demonstrated instance uses maximum-entropy RL strategies, an opponent classifier and a reviewer that can replace strategies over time.

**Connection:** The project uses customer profile, emotion and objection information to make action selection depend on context.

**Difference:** The current implementation has one PPO action policy. It does not implement the paper's mixture of strategies, deep opponent classifier or reviewer module.

**Useful oral explanation:** "It motivates adapting the conversational action when the customer's behavior changes."

### Study 3: General Negotiation Strategies

**Paper:** Bram M. Renting, Thomas M. Moerland, Holger H. Hoos and Catholijn M. Jonker, *Towards General Negotiation Strategies with End-to-End Reinforcement Learning*, RLC 2024. [Primary source](https://arxiv.org/abs/2406.15096).

The paper represents observations and actions as a graph and uses graph neural networks in its RL policy. This addresses negotiation problems whose observation and action dimensions vary.

**Connection:** It identifies a meaningful future direction for generalization across different product domains or action sets.

**Difference:** This project fixes its observation at 94 features and its action space at 16 actions. Changing these dimensions would require model and checkpoint changes. It has no implemented graph policy.

**Useful oral explanation:** "The paper helps explain why my current fixed representation is suitable for a bounded prototype but limits transfer to substantially different negotiation problems."

### Study 4: Persuasion Games with LLMs

**Paper:** Shirish Karande, Santhosh V and Yash Bhatia, *Persuasion Games with Large Language Models*, ICON 2024, pp. 576-582. [Primary source](https://aclanthology.org/2024.icon-1.67/).

The paper presents a multi-agent persuasion framework. A primary agent converses with the user, while auxiliary agents handle tasks including retrieval, response analysis, strategy development and fact validation. Its applications explicitly include insurance, and it evaluates simulated personas.

**Connection:** It motivates treating response content and fact validation as distinct responsibilities in an insurance dialogue system.

**Difference:** This project assigns conversational action selection to PPO and uses a structured product catalogue. It does not reproduce the paper's complete multi-agent framework.

**Correction to remember:** The venue is ICON 2024. Hosting on ACL Anthology does not make it an ACL 2024 conference paper.

### Study 5: AgenticPay

**Paper:** Xianyang Liu, Shangding Gu and Dawn Song, *AgenticPay: A Multi-Agent LLM Negotiation System for Buyer-Seller Transactions*, arXiv preprint, 2026. [Primary source](https://arxiv.org/abs/2602.06008).

AgenticPay provides a benchmark and simulation framework for natural-language buyer-seller negotiation. It covers bilateral and more complex market settings and evaluates agreement feasibility, efficiency and welfare.

**Connection:** It motivates evaluating complete multi-turn behavior and including a genuine direct-LLM policy baseline.

**Difference:** The project uses its own insurance simulator. The original saved benchmark skipped the direct-LLM experiment because a model was unavailable then.

**Correction to remember:** AgenticPay does not establish the specific seven-agent insurance architecture described in the review's original mechanism column.

### Research Position

These studies inform different parts of the design. Their findings do not directly establish performance for this project. The project investigates whether a relatively compact RL policy, an explicit product engine and controlled response generation work together in a bounded insurance sales setting.

The clearest research gaps to discuss are transfer to real buyer language, reliable estimation of customer state, sensitivity to the reward design, and evaluation against LLM policies and human judgments.

### Transition

> Based on these ideas, I implemented three main responsibilities: action selection, product validation and response generation.

## Slide 4: Implemented Approach

### Speaking Script

> The policy selects a discrete action such as clarification, product explanation or objection handling. The product engine filters information using need, age and budget. A local Llama model through Ollama then expresses the assigned strategy using the supplied facts. Verified templates provide a fallback. The live conversation layer also handles explicit requirements, such as clarifying an ambiguous number or explaining that an exact premium is unavailable. The decision trace records the policy proposal and any override. This separation makes the action and wording easier to inspect during the demonstration.

### Responsibilities

| Component | Input | Output | Purpose |
| --- | --- | --- | --- |
| Understanding | Buyer text | Intent, emotion, objection and extracted entities | Identify the current request |
| State builder | Understanding, profile and context | Structured state and 94-feature vector | Give the policy a compact observation |
| PPO policy | Encoded state and valid-action mask | An action index | Choose a next conversational action |
| Action-to-strategy mapping | Selected action | Controlled strategy label | Give generation an explicit purpose |
| Product engine | Need, age and budget | Eligible synthetic product facts | Restrict available recommendations |
| Generator | Strategy, facts and recent dialogue | Candidate response | Produce wording |
| Response checks | Candidate and allowed context | Accept/reject information and diagnostics | Detect supported classes of response errors |
| Conversation memory | Completed turns | Recent dialogue and compact summary | Preserve context for later turns |

### Action Versus Strategy Versus Response

An **action** is the policy's discrete choice. For example, `HANDLE_OBJECTION`.

A **strategy** is the controlled instruction given to generation. For that action, the mapping produces `OBJECTION_HANDLING`.

A **response** is the actual sentence shown to the buyer. Different sentences can express the same strategy. The RL policy learns probabilities over actions, while the LLM generates words.

### The LLM's Role

The local demonstration uses `llama3.2:3b` through Ollama. It receives the assigned strategy, approved context, eligible facts and recent conversation. The response uses a constrained JSON format, followed by validation.

PPO training in this project does not fine-tune Llama's weights. Improving the strategy policy and improving the language model are different experiments. The original research benchmark used the available template-based generation path.

### What the Product Engine Knows

The catalogue contains eight synthetic products. Records include a product identifier, coverage, sum insured, age range, premium band, waiting period, exclusions, benefits, conditions and source/version information.

Its premium values are bands such as `low`, `mid` and `high`. These do not specify an annual or monthly payment. A coverage amount such as `5L` describes cover in the synthetic catalogue and must not be presented as the premium.

Filtering supports the implemented need, age and budget rules. It does not perform complete insurer underwriting. Unknown budget currently permits multiple premium bands, so the next response may still need clarification.

### What the Checks Can Detect

The implementation checks numeric claims against the allowed facts, certain unsupported guarantees, recent literal repetition, buyer-role wording and selected dialogue requirements. It also verifies the requested strategy format. A failing or unavailable LLM response can trigger a template fallback with a recorded reason.

These checks have a defined coverage. They do not establish the truth of every possible paraphrase or imply complete semantic validation.

### Transition

> The policy and understanding models need data. The current experiment uses a versioned synthetic dataset with grouped train, validation and test splits.

## Slide 5: Dataset and Customer Understanding

### Speaking Script

> The dataset contains 1,200 synthetic conversations and 20,396 buyer and seller turns, along with 3,000 synthetic profiles. I group related scenarios and identical transcripts before splitting them to reduce leakage. The actual conversation split is 812 training, 164 validation and 224 test. Four Naive Bayes classifiers predict intent, emotion, objection and stage, while rules extract entities. Emotion accuracy is 53.7 percent and stage accuracy is 37 percent on the held-out synthetic set. These results identify understanding as a major area for improvement, particularly before evaluating real customer conversations.

### Dataset Construction Workflow

```text
Define insurance needs, profiles, objections and stages
    -> Generate annotated synthetic dialogue scenarios
    -> Normalize records and attach provenance
    -> Validate labels, ranges and action/strategy consistency
    -> Group scenario variants and identical transcripts
    -> Split groups into train, validation and test
    -> Train models and save evaluation reports
```

| Dataset item | Recorded size or property |
| --- | --- |
| Conversations | 1,200 |
| Dialogue turns, including both speakers | 20,396 |
| Customer profiles | 3,000 |
| Training conversations | 812 |
| Validation conversations | 164 |
| Test conversations | 224 |
| Held-out NLP test examples | 1,848 |
| Source in the default run | Entirely synthetic |
| Dataset version | v2 |

The split counts sum to 1,200. The target ratios are approximately 70/15/15, but indivisible groups produce the actual counts above. Splitting individual turns would let near-identical conversation context appear in both training and testing.

### Labels and Entities

The annotated taxonomy contains nine intent classes, eight emotion classes, nine need categories, eleven objection categories including `NONE`, eleven stages, sixteen actions and sixteen mapped strategies.

Examples of extracted entities include insurance need, age, budget band and existing cover. A bare amount such as `250000` needs a meaning: cover, premium budget, income or something else. The live layer stores an ambiguous amount without converting it into a budget band.

### Why Naive Bayes?

Multinomial Naive Bayes provides a small, fast text-classification baseline suitable for the current environment. It estimates a class from the frequency of words associated with that class in training examples. Separate classifiers address the different prediction tasks.

It has limited ability to capture complex context, implied meaning and emotion. A stronger language encoder could be evaluated later, but would need meaningful data and a comparison under the same test conditions.

### Held-Out NLP Results

| Task | Accuracy | Macro-F1 | Interpretation |
| --- | ---: | ---: | --- |
| Intent | 100.0% | 1.000 | Repetitive synthetic wording makes this an easy test |
| Emotion | 53.7% | 0.515 | Emotion understanding remains limited |
| Objection | 100.0% | 1.000 | Synthetic separation does not establish free-form accuracy |
| Stage | 37.0% | 0.306 | Single-utterance stage prediction remains weak |

Accuracy measures the fraction of correct labels. Macro-F1 gives each class equal weight and summarizes precision and recall. Generated emotion and stage labels do not necessarily have strong linguistic evidence in the utterance.

### Important Implementation Details

The four classifiers have recorded offline evaluations. In the live conversation flow, however, the stage comes from conversation rules using stop requests, turn count, known need and objections. Do not imply that the measured stage classifier drives every live stage transition.

The live layer also uses explicit cues for certain product and payment requests. It retains the original NLP predictions in `raw_understanding` and records whether an output came from a model or an explicit cue.

Public dataset readers exist for MultiDoGO, GoEmotions and MELD, but the saved default experiment did not load those corpora. The dataset and performance figures above belong to the project's synthetic run.

### Transition

> The next diagram shows how these components operate during a buyer conversation and during offline learning.

## Slide 6: System Pipeline

### Speaking Script

> In a live turn, the buyer's text goes through understanding and entity extraction. The state builder combines that information with the profile and context. PPO selects an allowed action, and explicit conversation rules can adjust it for a known task. The product engine supplies eligible facts. The generator then produces a checked response, and memory carries context into the next turn. During training, a customer simulator produces the next state and reward. PPO collects these transitions and updates its parameters. Live chat uses the saved checkpoint and does not learn from each buyer message.

### Live Inference Workflow

1. **Receive text.** Check that the message is non-empty and the session is open.
2. **Predict understanding.** Run the NLP models and entity extractor.
3. **Apply explicit cues.** Identify tasks such as initial product discovery, a payment question, an ambiguous amount or rejection.
4. **Update session facts.** Remember the need, budget, existing cover, age and amount context when available.
5. **Construct state.** Build the structured state and encode the 94-dimensional observation.
6. **Compute eligible products and action mask.** Restrict product actions if relevant information or eligible products are unavailable.
7. **Select an action.** The live PPO path chooses the highest-probability valid action from the saved model. A rule-policy option is also available.
8. **Apply a live task constraint where necessary.** Record the original policy action and the decision source.
9. **Map the action to a strategy.** Supply its purpose to generation.
10. **Generate and check a response.** Use the configured LLM or templates and record response source or rejection diagnostics.
11. **Update memory.** Save the completed buyer-agent turn and previous action.
12. **Close when required.** Respecting an explicit rejection ends the session.

### Training Workflow

```text
Supervised policy weights
    -> Initialize PPO actor
    -> Reset customer simulator
    -> Encode current state and compute valid actions
    -> Sample an action from the masked policy
    -> Apply action in the environment
    -> Observe reward, next state and termination flags
    -> Store rollout transition
    -> Compute GAE and critic targets
    -> Update actor and critic over minibatches
    -> Collect the next rollout
    -> Save checkpoint, training history and evaluation outputs
```

### What the Simulator Does

`CustomerSim` maintains a synthetic personality, need, budget, age, existing cover, objection, patience and behavioral scores. Its stochastic rules determine reactions to actions.

For example, a price-sensitive profile has preferences for affordability framing, alternatives and value framing. A skeptical profile favors trust-building, comparison and product explanation. Positive or negative reactions modify simulated trust, satisfaction, engagement and purchase intent.

These preferences are hand-designed assumptions. The simulator is useful for controlled experiments, but a policy can become effective at its assumptions without becoming effective with people.

The transition model responds primarily to the selected action and product suitability. It does not semantically judge the generated sentence. Therefore changing LLM wording does not automatically test the wording's effect on buyer behavior.

### Worked Buyer Example

| Turn | Buyer message | System interpretation | Appropriate demonstrated behavior |
| --- | --- | --- | --- |
| 1 | "I need health insurance for my family." | Family-health product request | Ask about cover or existing insurance |
| 2 | "around 250000" | Amount with unknown meaning and payment period | Ask whether it refers to coverage or a premium budget |
| 3 | "what will be my payment" | Payment inquiry | Explain that the catalogue lacks an exact premium quote |
| 4 | "I am not interested. Please stop." | Explicit stop request | Respect rejection and close the session |

The first three messages correspond to the tested demonstration flow. The fourth is a suggested rejection check. Exact wording can vary across LLM runs.

When describing the trace, identify whether an action came from PPO or a conversation guard. The amount-clarification and payment behavior includes explicit live constraints.

### Transition

> To train the policy, I express the dialogue environment in terms of state, action, transition and reward.

## Slide 7: Reinforcement Learning Formulation

### Speaking Script

> The environment exposes a 94-feature observation and sixteen discrete actions. The state includes intent, emotion, need, objection, stage, profile and previous action, together with normalized behavioral values and coverage information. The actions range from discovery and education to objection handling and respectful rejection. Action masks remove choices that are invalid for the current context. The reward gives positive feedback for useful discovery, engagement, suitable recommendations and purchases. It penalizes pressure, unsupported claims and unsuitable recommendations. This produces a multi-objective learning problem. A simulated purchase has the largest positive event reward, but useful conversation behavior can also accumulate reward.

### Elements of the RL Problem

| RL concept | Meaning in this project |
| --- | --- |
| Agent | The conversational action policy |
| Environment | Insurance environment wrapping the synthetic customer simulator |
| Observation | Encoded customer and conversation state |
| Action | One of sixteen conversational actions |
| Transition | The simulator's reaction and state update |
| Reward | Weighted events following the action |
| Episode | A conversation ending in a terminal outcome or time limit |
| Policy | A probability distribution over valid actions given the observation |
| Objective | Maximize expected discounted cumulative reward |

Conceptually, the interaction is `s_t -> a_t -> reward_t, s_(t+1)`.

The finite state is an engineering approximation. Real buyer beliefs are partly hidden and may depend on earlier context that the vector omits. The project has no implemented belief-state or recurrent policy to solve that partial observability problem.

### Exact 94-Feature Breakdown

| Feature group | Dimensions | Encoding |
| --- | ---: | --- |
| Intent | 9 | One-hot |
| Emotion label | 8 | One-hot |
| Need | 9 | One-hot |
| Objection | 11 | One-hot |
| Sales stage | 11 | One-hot |
| Customer profile | 8 | One-hot |
| Previous action | 16 | One-hot |
| Behavioral values, turn, budget and coverage gap | 8 | Bounded scalar values |
| Emotion probabilities | 8 | Normalized probability vector |
| Existing coverage | 3 | Indicators |
| Previous reaction | 3 | Indicators |
| **Total** | **94** | **72 + 8 + 8 + 3 + 3** |

The eight scalar features are purchase intent, trust, engagement, satisfaction, objection severity, normalized turn count, encoded budget and a coverage-gap indicator.

Age is available to product filtering but is not a separate coordinate in the 94-feature vector. Raw text, the full transcript and the numeric `amount_context` are also outside this vector. They can still affect extraction, conversation constraints and generation.

### State Limitations Worth Knowing

In simulation, the environment knows its synthetic behavioral values. In live chat, trust, engagement and satisfaction currently use defaults of 0.5 and purchase intent uses 0.3. They are not calibrated measurements of the real buyer.

The live call also leaves `previous_reaction` at its default rather than deriving the simulator's positive/neutral/negative reaction label. Its three reaction indicators therefore do not carry the same information as in simulated training.

Unknown and mid budget both encode as 0.5, although the structured session retains the label. This loses a distinction at the policy-vector level. These differences help explain why simulator performance does not transfer automatically to live interaction.

### Complete Action Space

| Action | Purpose | Example conversational intention |
| --- | --- | --- |
| `DISCOVER_NEEDS` | Learn what protection the buyer needs | Ask who needs cover |
| `ASK_CLARIFYING_QUESTION` | Resolve missing or ambiguous information | Ask what an amount refers to |
| `DISCOVER_COVERAGE_GAP` | Understand existing protection | Ask about current coverage |
| `EXPLAIN_PRODUCT` | Explain an eligible product | Describe supplied benefits and exclusions |
| `VALUE_FRAMING` | Relate benefits to an identified need | Explain relevance to the household |
| `RISK_EXPLANATION` | Explain relevant protection needs | Discuss the risk category without invented claims |
| `AFFORDABILITY_FRAMING` | Discuss budget constraints | Explain available premium bands |
| `HANDLE_OBJECTION` | Address a stated concern | Clarify a coverage objection |
| `YES_YES_FRAMING` | Acknowledge a valid concern and suggest a constructive next step | Agree that unnecessary cover should be avoided |
| `BUILD_TRUST` | Provide transparent information | Explain limitations and known facts |
| `PERSONALIZE` | Relate an option to known buyer context | Discuss an eligible family-health option |
| `COMPARE_OPTIONS` | Compare approved facts | Compare eligible catalogue entries |
| `OFFER_ALTERNATIVE` | Present another eligible choice | Consider a different premium band when permitted |
| `ASK_FOR_COMMITMENT` | Ask whether the buyer wishes to proceed | Request a voluntary next step |
| `FOLLOW_UP` | Allow further consideration | Offer a later discussion |
| `RESPECT_REJECTION` | End selling after rejection | Close politely |

"Yes-yes" in this project means acknowledging a legitimate concern, agreeing where appropriate and moving toward a relevant solution. It is not implemented as a requirement that the buyer agree or purchase.

### Action Masking

The policy computes probabilities only over allowed actions. Invalid logits are set to negative infinity before softmax, making their probabilities zero.

The mask blocks product actions before the need is known or when no eligible product is available. It blocks early commitment. An explicit stop request or rejection stage leaves only `RESPECT_REJECTION` available. The live initial-discovery task further restricts the candidate set to appropriate discovery actions.

Masks constrain available behavior. Reward penalties influence what the policy learns among available choices. They serve different purposes.

### Full Configured Reward Table

| Event or configuration key | Weight |
| --- | ---: |
| Need discovery | +0.2 |
| Useful clarification | +0.1 |
| Objection resolution | +0.3 |
| Engagement improvement | +0.3 |
| Trust improvement | +0.3 |
| Suitable product action | +0.5 |
| Qualified-lead threshold crossing | +1.0 |
| Suitable purchase | +5.0 |
| Repetition | -0.3 |
| Premature closing component | -0.5 |
| Pressure | -1.0 |
| Unsupported claim | -2.0 |
| Unsuitable recommendation | -3.0 |
| Negative reaction / frustration | -1.0 |

Multiple events can apply to one step. For example, a suitable product explanation with improved engagement and trust can receive `0.5 + 0.3 + 0.3 = 1.1`, if no other events apply. This is a worked calculation, not a measured trajectory.

There is a code detail beyond the configuration labels: a `premature` event adds `0.5 * pressure_weight + premature_closing_weight`. With the default weights, that contribution is `-1.0`. A separate pressure event can add another penalty. The slide shows selected configuration weights, so use the code-level explanation if asked about the exact total.

The lead reward applies when simulated purchase intent crosses above 0.6 from at most 0.6. A simulated purchase requires purchase intent above 0.8, a valid commitment action and an eligible option. These thresholds are modeling choices.

### Episode Ending

The environment's maximum length is twenty turns. Purchase and respectful rejection create terminal outcomes. Reaching the time limit without a terminal outcome creates truncation. The PPO implementation treats these differently when estimating future value.

### Transition

> PPO uses this environment to collect experience and update both its action policy and its value estimator.

## Slide 8: PPO Implementation in Detail

### Speaking Script

> PPO uses an actor to choose actions and a critic to estimate future reward. My implementation is a portable linear actor-critic in NumPy. Supervised action weights initialize the actor. During training, the agent samples valid actions and saves their probabilities with the observed rewards. Generalized advantage estimation then measures whether an action performed better or worse than expected. PPO compares the new action probability with the probability recorded during the rollout and uses a clipped objective to discourage excessively large improvements on the same data. I use Adam, entropy regularization and gradient clipping. Each seed trains for twenty thousand steps. The plotted curve shows actual rolling training reward for seed forty-two, including its fluctuations.

### Why PPO Fits This Experiment

The action space is discrete, and success depends on a sequence of decisions. PPO can learn from simulator rollouts while a value function helps estimate longer-term consequences. Its clipped policy objective supports multiple minibatch updates on a collected rollout.

This is an algorithm choice, not a result that PPO must outperform every alternative. The project tests the choice against simpler baselines. The recorded comparison shows a tradeoff between composite reward and conversion.

### Actor and Critic

The actor computes linear action scores from the state:

```text
logits = state @ W + b
policy = masked_softmax(logits)
```

The actor weight matrix has shape `94 x 16`. The critic is a separate linear function:

```text
V(state) = state @ value_weights + value_bias
```

The actor chooses an action. The critic estimates expected future discounted reward. The critic does not generate a reply or decide whether a policy claim is true.

### Supervised Initialization

The supervised baseline learns from synthetic action labels. Its compatible action weight matrix initializes the PPO actor in the full configuration. PPO then updates the actor using rewards from new simulator interactions. The critic begins separately.

This provides an initial action preference. The `no_supervised_init` ablation tests training without that initialization. Warm starting is an experimental design choice whose benefit must be measured.

### Rollout Collection

A rollout stores observations, actions, old log probabilities, rewards, values, next-state values, action masks and episode-boundary flags. Training samples stochastically so the agent explores valid actions. Live inference and the implemented PPO evaluation policy use deterministic action selection.

The saved log probability must come from the same masked distribution used to sample the action. Otherwise, the probability ratio used by PPO would compare inconsistent distributions.

### Generalized Advantage Estimation

The advantage estimates how much better an observed action performed than the critic expected.

The implementation computes a temporal-difference residual:

```text
delta_t = reward_t
          + gamma * V(next_state) * (1 - terminated_t)
          - V(state_t)
```

It then accumulates advantages backward:

```text
A_t = delta_t
      + gamma * lambda * (1 - boundary_t) * A_(t+1)
```

Here `boundary_t` is true for termination or truncation. The critic target is:

```text
return_target_t = A_t + V(state_t)
```

The actor uses normalized advantages. The critic uses the raw return targets. Normalizing critic targets with the actor advantages would change their scale and meaning.

At a true terminal outcome, there is no future episode value to bootstrap. At a time limit, the environment may still have future value, so the implementation bootstraps `V(next_state)` but stops advantage propagation across the reset.

### PPO Probability Ratio and Clipping

To avoid confusion with reward, call the probability ratio `rho` in an explanation:

```text
rho_t = pi_new(action_t given state_t)
        / pi_old(action_t given state_t)

L_clip = mean(min(rho_t * A_t,
                 clip(rho_t, 1 - epsilon, 1 + epsilon) * A_t))
```

The slide uses `r` for this same ratio. With `epsilon = 0.2`, the clipping interval is `[0.8, 1.2]`.

For a positive advantage of `1.0`, suppose an action probability changes from `0.20` to `0.30`. The ratio is `1.5`. The clipped term uses `1.2`, limiting the incentive to keep increasing that action probability on this example.

Clipping modifies the optimization objective. It is not a hard guarantee that every resulting probability remains within twenty percent of its old value. Shared parameters, minibatch updates and other objective terms can change probabilities further.

### Other Optimization Components

**Entropy regularization** encourages a less concentrated action distribution during training and supports exploration.

**Critic fitting** reduces squared error between predicted value and the raw return target.

**Adam** maintains running gradient statistics to update the actor and critic parameters.

**Gradient clipping** scales the joint gradient when its norm exceeds the configured limit. This controls the magnitude of an update's gradient, independently of PPO's probability-ratio clipping.

### Actual Hyperparameters

| Parameter | Value | Meaning |
| --- | ---: | --- |
| Training timesteps | 20,000 per run | Simulator actions collected |
| Rollout length | 1,024 | Maximum transitions before an update |
| Minibatch size | 256 | Transitions per minibatch |
| Epochs per rollout | 4 | Passes over the collected rollout |
| Learning rate | 0.0003 | Adam update step scale |
| Discount factor, gamma | 0.99 | Weight of future reward |
| GAE lambda | 0.95 | Advantage-estimation parameter |
| PPO clipping range | 0.2 | Probability-ratio clipping parameter |
| Entropy coefficient | 0.01 | Exploration regularization strength |
| Maximum gradient norm | 0.5 | Joint gradient limit |
| Training seeds | 42, 43, 44 | Independent PPO runs |

The final rollout can be shorter than 1,024 transitions so the run ends at exactly 20,000 steps.

### Why NumPy Instead of Stable Baselines3?

In this environment, Windows Application Control prevents Torch's `torch_python.dll` from loading. The recorded experiment therefore uses the NumPy implementation. It is a linear reference backend with explicit PPO calculations. Benchmarking against a maintained RL implementation remains useful future validation when the environment supports it.

Do not describe this checkpoint as a trained deep neural PPO model or an SB3 result.

### How to Explain the Training Curve

The chart comes from the saved seed-42 training history. Each point shows mean reward over the latest one hundred completed training episodes at a rollout update. Its horizontal axis is training timesteps.

The curve fluctuates and finishes around four reward units. It does not show monotonic improvement or prove convergence. Generalization should be assessed with held-out evaluation, which appears on the next slide.

### Transition

> I evaluated the learned policy against three simpler baselines using held-out simulated scenarios.

## Slide 9: Evaluation Results and Interpretation

### Speaking Script

> Each baseline uses three seed runs and two hundred held-out simulated episodes per seed. PPO achieves a mean reward of 4.12 compared with 3.86 for the rule policy. However, simulated conversion is 5.17 percent for PPO and 12.33 percent for the rule policy. These metrics favor different policies because the reward also values engagement, trust, objection handling and suitable recommendations. PPO's objection-resolution rate is 78.2 percent, compared with 69.7 percent for the rule policy. The intervals across three seeds are wide, so I present these as preliminary findings. Six independently retrained ablations examine the state features, action set, reward and initialization.

### Baselines

| Policy | How it selects an action | What it helps test |
| --- | --- | --- |
| Random | Samples from valid actions | Performance available from constraints and chance |
| Rule | Uses hand-written decisions based on objection/stage context | Whether learning improves on domain heuristics |
| Supervised | Predicts a labeled action from synthetic demonstrations | What imitation alone achieves |
| PPO | Uses the learned masked policy | What reward-based sequential training adds |
| Direct LLM | Asks an LLM to choose an allowed strategy | A planned comparison available in code but skipped in the saved run |

All policies use the same product database, turn limit and evaluation scenario-seed schedule. Different actions can consume random draws differently, so complete trajectories need not stay identical after their initial conditions match.

### Recorded Baseline Metrics

| Policy | Mean reward | Reward SD | Conversion | Qualified leads | Objection resolution | Satisfaction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | 2.6222 | 0.2101 | 0.33% | 36.67% | 61.09% | 0.6204 |
| Rule | 3.8602 | 0.5114 | 12.33% | 47.17% | 69.70% | 0.6466 |
| Supervised | 3.6260 | 0.5826 | 0.00% | 41.67% | 51.27% | 0.6456 |
| PPO | 4.1195 | 0.6833 | 5.17% | 48.33% | 78.20% | 0.6692 |

Percentages above convert the saved, rounded CSV proportions to percent. The slides use fewer decimal places for readability. Each metric summarizes three seed means.

### Metric Definitions

**Mean reward:** Average cumulative environment reward per episode.

**Conversion:** Fraction of episodes ending with the simulator's purchase condition.

**Qualified leads:** Fraction of episodes with final simulated purchase intent above 0.6. This final-state metric differs from the threshold-crossing event used for the lead reward during a trajectory.

**Objection resolution:** Resolution measured over episodes that initially contain an objection. Episodes with no initial objection do not enter that denominator.

**Satisfaction:** The simulator's final bounded satisfaction score. It is not a human survey result.

**Violation counters:** Counts accumulated over conversation turns for the implemented pressure, unsupported-claim and unsuitable-recommendation signals.

### Why Can PPO Earn More Reward but Convert Less?

The reward has several components. A policy can earn reward through trust and engagement improvements, objection resolution and suitable product actions without frequently reaching a purchase. The rule policy may choose commitment in ways that create more conversions under the simulator's rules.

The metrics establish the difference, but they do not isolate its complete cause. Detailed action frequencies and trajectories would help test explanations. Avoid claiming that the PPO policy is automatically more ethical or that it deliberately sacrificed sales to protect customers.

PPO exceeds the rule baseline by about `0.2593` mean reward units, while its conversion rate is about `7.16 percentage points` lower using the saved values. Objection resolution is `8.50 percentage points` higher. These are descriptive differences.

### Uncertainty

| Policy | 95% interval for mean reward across seeds |
| --- | --- |
| Random | 2.1003 to 3.1440 |
| Rule | 2.5896 to 5.1308 |
| Supervised | 2.1786 to 5.0734 |
| PPO | 2.4219 to 5.8171 |

The implementation uses a Student-t interval over seed means. With three seeds, uncertainty is substantial. The recorded analysis does not establish statistical superiority or performance with real customers. A suitable next study would increase independent runs and examine paired differences and effect sizes.

Some saved intervals for bounded proportions extend below zero or above one because the simple t-interval is not constrained to the probability range. Such bounds are an artifact of this interval method, not negative probabilities. More suitable uncertainty estimates for bounded metrics should be considered in a later analysis.

### Ablation Experiments

An ablation changes a component and retrains the policy. Simply hiding a feature from an already-trained policy would measure a different question.

Each listed configuration uses the same training-step budget and three seeds, with one hundred evaluation episodes per seed. Reward variants are evaluated using the shared original reward.

| Configuration | Mean reward | Conversion | Objection resolution |
| --- | ---: | ---: | ---: |
| Full PPO, ablation evaluation batch | 4.3927 | 5.33% | 73.68% |
| No emotion features | 4.1210 | 0.00% | 89.57% |
| No objection features | 4.1883 | 2.33% | 54.96% |
| No yes-yes action | 4.3100 | 2.33% | 74.04% |
| Reduced action set | 4.4133 | 4.33% | 71.65% |
| Conversion-only training reward | 4.3690 | 9.00% | 82.00% |
| No supervised initialization | 3.9540 | 9.00% | 94.97% |

The full-PPO row here comes from a separate evaluation batch with a different episode budget. Compare an ablation to that row, rather than substituting its value for the baseline table's 4.1195.

The findings are mixed. Removing emotion reduces mean reward and conversion in this batch but increases objection resolution. Removing objections reduces objection resolution. The no-initialization configuration has a large reward SD of 2.7969. These results support further investigation, not a claim that every feature improves every metric.

### What Was Not Measured in the Saved Benchmark?

The direct-LLM policy baseline and LLM wording ablation were skipped because the original run had no configured reachable model. Installing and demonstrating Llama later does not add those missing experiments retroactively.

The recent live dialogue fixes did not retrain PPO or rerun the full research benchmark. The displayed simulator results remain the recorded earlier experiment.

### Transition

> The final slide shows the current demonstration and the work that remains before a stronger research conclusion.

## Slide 10: Demonstration and Remaining Work

### Speaking Script

> The local interface supports buyer profiles, PPO or rule policies, and either the local LLM or template responses. It also shows the selected action, interpreted state, response checks and source. The demonstrated payment question now leads to an explanation that exact premiums require an insurer quote. At this interim stage, the dataset, simulator, trained PPO, baseline results and text interface are implemented. My next priorities are stronger emotion and stage understanding, broader free-form testing, and the direct-LLM and human evaluations. Voice can follow as an additional interface. The current catalogue remains synthetic, and policy issuance and payment processing are outside the implemented demo.

### Demonstration Sequence

1. Open the local UI and choose the family-oriented profile with a known age, such as 35.
2. Select PPO and the desired response engine.
3. Send: `I need health insurance for my family.`
4. Explain the interpreted need, selected action and response source in the decision trace.
5. Send: `around 250000`.
6. Explain why the amount requires clarification and show its unknown meaning in the trace.
7. Send: `what will be my payment`.
8. Explain that the synthetic catalogue contains premium bands, so it cannot supply an exact insurer quote.
9. If time permits, test rejection in the same session or a fresh session.
10. Use the Results view to connect the working interface to the saved experiments.

In the recorded three-turn local LLM smoke test, responses took approximately 31 to 45 seconds per turn. That is one hardware-specific observation, not a latency benchmark. For a ten-minute presentation, use the screenshot to explain the flow and run live inference only if it fits the available time. Template mode demonstrates the system without waiting for language generation.

### Starting the Demonstration

From PowerShell:

```powershell
Set-Location 'D:\insurence seller\insurance_sales_agent'
& 'D:\insurence seller\term_project\Scripts\python.exe' demo_server.py
```

Open [the local demo](http://127.0.0.1:8765/). If that port is already serving this project, reuse it. To start a separate instance on another port:

```powershell
& 'D:\insurence seller\term_project\Scripts\python.exe' demo_server.py --port 8766
```

The UI server can start the project-local Ollama executable when needed. Local LLM mode requires the installed model to become available. Template mode works without it. There is no need to rerun model training merely to present the saved checkpoint.

### Current Completion Status

| Area | Current status |
| --- | --- |
| Synthetic data and provenance | Implemented |
| Grouped data splitting | Implemented |
| Four NLP classifiers and entity extraction | Implemented, with limited emotion/stage accuracy |
| State representation and memory | Implemented |
| Synthetic product eligibility | Implemented |
| Customer simulator and reward | Implemented |
| PPO training and checkpointing | Implemented |
| Random, rule, supervised and PPO evaluation | Recorded over three seeds |
| Six retrained ablations | Recorded |
| Local text UI and Llama wording | Demonstrated |
| Direct-LLM benchmark and wording ablation | Available as experiment paths, missing from the original results |
| Public-corpus transfer study | Pending |
| Human evaluation | Pending |
| Speech recognition and text-to-speech | Pending |
| Genuine insurer quote, issuance and payments | Not implemented |

### Next Research Priorities

**Improve state estimation.** Evaluate stronger emotion and stage models with more varied and appropriately annotated text. Make training observations resemble the uncertainty and missing information of live use.

**Strengthen policy evaluation.** Run more independent seeds, compare with a maintained RL implementation when supported, and analyze action distributions and reward components.

**Complete LLM comparisons.** Evaluate a direct-LLM action policy and separately test response wording. Record failures, latency and actual resource requirements.

**Assess human interaction.** Use a defined evaluation protocol for relevance, factuality, suitability, pressure and user understanding. Obtain appropriate data permissions before using real conversations.

**Extend the interface.** Add speech recognition before the existing text pipeline and text-to-speech after response generation. Voice adds transcription errors and latency that require their own evaluation.

### Closing Statement

> The current project demonstrates an end-to-end experimental framework for adaptive insurance sales dialogue. PPO achieved higher mean composite reward in the simulator, while the rule policy achieved higher conversion. These initial results make the reward tradeoff and the simulation-to-live gap clear, and they define the next experiments.

## Questions Faculty May Ask

### 1. What exactly is learned through reinforcement learning?

The actor's mapping from the encoded state to action probabilities and the critic's value estimate. Product rules and the LLM's pretrained weights are not learned by this PPO training loop.

### 2. Why use RL when a rule system already works?

Rules provide a strong baseline. RL can optimize action sequences from feedback, but its usefulness must be demonstrated. In these results, PPO has higher mean reward, while the rule policy has higher conversion.

### 3. Why not let the LLM choose everything?

An explicit action policy gives the experiment a controlled action space and a measurable learning objective. A separate product layer restricts available facts. A direct-LLM action policy is a relevant baseline, and that comparison remains to be completed.

### 4. Is this reinforcement learning from human feedback?

No. PPO learns from a hand-designed reward in a synthetic customer environment. There is no human preference dataset or trained human-feedback reward model in the current implementation.

### 5. Is Llama trained on your insurance dataset?

The current local Llama model is prompted with strategy, facts and context. This project has not fine-tuned its weights. The synthetic data trains the smaller understanding and action-selection components.

### 6. Does it learn continuously during the live demonstration?

No. It updates conversation memory and state, but it uses a saved PPO checkpoint. Policy optimization occurs in the offline training workflow.

### 7. Is the agent a deep RL system?

The current backend uses linear actor and critic models in NumPy. Some literature uses deep RL, but the saved project checkpoint should be described accurately as linear PPO.

### 8. Why are there 94 state dimensions?

The vector combines 72 categorical dimensions, eight scalar values, eight emotion probabilities, three coverage indicators and three reaction indicators. The exact encoding appears in `src/state_products.py`.

### 9. Does the policy see the whole conversation?

It receives the encoded vector. Recent dialogue and a memory summary also support the conversation layer and generator, but the full transcript is not directly an input to the linear PPO actor.

### 10. How is age handled?

Age is stored in the structured context and used by the product eligibility filter. It is not a separate feature in the 94-dimensional policy vector. Its effect can reach action selection through product availability and the mask.

### 11. Why include emotion if emotion accuracy is low?

The specification investigates emotion-aware action selection. Low classifier accuracy is an important limitation, and the emotion ablation tests the feature's contribution within the simulated experiment. A better live emotion model and a transfer evaluation are needed.

### 12. Is simulated emotion the same as live predicted emotion?

No. The simulator derives emotion from its own reaction rules. Live emotion comes from the text model. This is one of the training-to-deployment distribution differences.

### 13. How did you prevent data leakage?

Scenario variants and identical normalized transcripts stay within the same split. This reduces direct leakage, while repeated synthetic wording still limits how challenging the test set is.

### 14. Why are intent and objection accuracy 100 percent?

The synthetic examples use repeated, strongly associated wording. The result describes that held-out synthetic dataset and does not imply perfect understanding of real messages.

### 15. Where do the rewards come from?

The environment computes events such as need discovery, an objection resolving, a threshold crossing or an unsuitable action. The reward function combines their configured weights. They are design choices rather than measured human preferences.

### 16. Is the reward fair or ethically optimal?

That has not been established. It includes penalties for pressure and unsuitable behavior, but validating the objectives and their weights requires domain review and human evaluation.

### 17. Can the policy exploit the reward?

Yes, that is a general experimental concern. Dense rewards for intermediate behavior can favor actions that score well without producing the desired overall outcome. Trajectory analysis, ablations and improved evaluation are needed to examine this.

### 18. Why is the rule policy's conversion higher?

The measured objective is broader than conversion. The rule policy's action choices reach the simulator's purchase condition more often. Identifying the exact causal mechanism requires inspecting its decisions and the PPO trajectories.

### 19. What does a five-percent conversion rate mean?

About five percent of the evaluated simulated conversations satisfy the environment's purchase condition, according to the rounded seed-mean result. It is not a percentage of actual buyers purchasing insurance.

### 20. What does zero reported violation mean?

The configured counters detected no such events in those constrained simulated runs. It does not demonstrate that every possible live LLM response will be factual or appropriate.

### 21. Is PPO statistically better than the rule baseline?

The current analysis does not establish that conclusion. The mean reward is higher, but the three-seed confidence intervals are wide. A more adequately powered comparison is needed.

### 22. What is the difference between PPO clipping and gradient clipping?

PPO clipping changes the action probability-ratio objective. Gradient clipping limits the norm of the parameter-update gradient. Both influence training, but they act on different quantities.

### 23. What is an ablation?

It is an experiment that removes or changes a component and retrains under a controlled budget. It helps test the contribution of that component to measured outcomes.

### 24. Did every ablation reduce performance?

No. Some metrics improve for some ablations. The mixed results and wide variability are part of the findings and should be reported directly.

### 25. Why is the direct-LLM row absent from the chart?

The original experiment could not run a configured local LLM and reported the baseline as skipped. The later live integration has not yet produced a replacement full benchmark.

### 26. Is the system negotiating insurance prices?

The current policy selects conversation strategies over a synthetic product catalogue. It does not have an insurer pricing engine or authority to negotiate premiums or claim settlements.

### 27. Why does it not give an exact payment for 250000?

The number's meaning is initially ambiguous. Even if it means desired coverage, the catalogue contains premium bands rather than exact rates. An actual premium requires a verified pricing source and relevant buyer details.

### 28. Can the current system sell a policy?

It can demonstrate a sales conversation and discuss eligible synthetic options. It has no real quotation, policy issuance or payment integration, so it cannot complete an actual insurance transaction.

### 29. Can it work with voice?

Voice is a planned extension. Speech recognition would convert audio to text before the existing pipeline, and text-to-speech would read the checked response. Neither component is implemented in this version.

### 30. What caused the earlier weak LLM responses?

Observed problems included buyer-role echoing, insufficient dialogue context, ambiguous amounts, task mismatch and repeated generic fallback text. The live flow now includes role-based history, explicit task constraints, compact prompts and more specific checks. A three-turn smoke test succeeded afterward, but it is not a broad quality evaluation.

### 31. Does a conversation guard weaken the RL contribution?

It means the live system is a combination of learned policy and explicit constraints. The trace makes that combination visible. Research claims must distinguish policy behavior from guard-driven behavior and evaluate the complete live flow separately.

### 32. What is the strongest contribution at the interim stage?

An inspectable end-to-end experimental implementation that combines an insurance state/action design, a simulator, constrained PPO and controlled wording, with recorded baselines and ablations. The next challenge is stronger evidence for generalization and usefulness.

## Quick Reference Before Presenting

| Topic | Phrase to remember |
| --- | --- |
| Project focus | Adaptive selection of insurance sales dialogue actions |
| Core division | PPO selects the action, rules filter facts, LLM expresses the response |
| Data | 1,200 synthetic conversations, 20,396 turns, 3,000 profiles |
| State/action dimensions | 94 features and 16 actions |
| Algorithm | Linear masked actor-critic PPO with GAE and supervised initialization |
| Training budget | 20,000 steps for each independent trained run |
| Evaluation | Three seeds, 200 baseline episodes per seed |
| Main result | PPO reward 4.12, rule reward 3.86 |
| Conversion comparison | PPO 5.17%, rule 12.33% |
| Important weakness | Synthetic-to-live state and language differences |
| Local LLM | Llama 3.2 3B through Ollama, with no insurance fine-tuning |
| Current interface | Text UI with decision trace and template fallback |
| Pending work | Broader evaluation, stronger NLP, direct-LLM tests, human testing and optional voice |

## Terminology

| Term | Plain-language explanation |
| --- | --- |
| Intent | What the buyer is trying to do with a message |
| Emotion | The model's estimate of the expressed emotional category |
| Objection | A concern or barrier to proceeding |
| State | The information available for a decision |
| Policy | A rule or learned function that selects an action |
| Actor | The learned action-probability model |
| Critic | The model estimating future reward |
| Return | Reward accumulated over future steps, with discounting |
| Advantage | How much better or worse an action was than expected |
| Rollout | A batch of experience collected by interacting with the environment |
| Action mask | A filter that makes invalid actions unavailable |
| Entropy | A measure of how spread out the policy's probabilities are |
| Warm start | Initialization using a previously learned model |
| Ablation | A controlled experiment changing one component |
| Inference | Using a saved model to make a prediction or decision |
| Checkpoint | Saved model parameters |
| Fallback | An alternative response path used when the main path is unavailable or fails checks |

## Source Map and References

### Local Project Evidence

| Subject | Source |
| --- | --- |
| Original research question and scope | [Project specification](<D:/insurence seller/Insurance_Sales_RL_Agent_Project_Specification.md>) |
| Supplied literature review | [AI negotiation research review](<D:/insurence seller/AI_Negotiation_Research_Insurance_v2.docx>) |
| Method and experimental limitations | [Methodology](<D:/insurence seller/insurance_sales_agent/docs/methodology.md>) |
| Data provenance and split | [Dataset card](<D:/insurence seller/insurance_sales_agent/docs/dataset_card.md>) |
| Taxonomies and action mapping | [Common definitions](<D:/insurence seller/insurance_sales_agent/src/common.py>) |
| State vector and catalogue | [State and products](<D:/insurence seller/insurance_sales_agent/src/state_products.py>) |
| NLP and extraction | [NLP implementation](<D:/insurence seller/insurance_sales_agent/src/nlp.py>) |
| Simulator, masks and exact reward | [Environment implementation](<D:/insurence seller/insurance_sales_agent/src/sim_env.py>) |
| PPO update and GAE | [NumPy PPO](<D:/insurence seller/insurance_sales_agent/src/ppo_numpy.py>) |
| Live state and action constraints | [Conversation implementation](<D:/insurence seller/insurance_sales_agent/src/conversation.py>) |
| LLM/template generation and checks | [Generation implementation](<D:/insurence seller/insurance_sales_agent/src/generation.py>) |
| Baselines and episode metrics | [Agents](<D:/insurence seller/insurance_sales_agent/src/agents.py>) |
| Seed aggregation and ablations | [Evaluation implementation](<D:/insurence seller/insurance_sales_agent/src/evaluate.py>) |
| Reward and PPO hyperparameters | [Base configuration](<D:/insurence seller/insurance_sales_agent/configs/base.yaml>) |
| Evaluation budgets | [Evaluation configuration](<D:/insurence seller/insurance_sales_agent/configs/evaluation.yaml>) |
| Baseline numerical results | [Baseline CSV](<D:/insurence seller/insurance_sales_agent/results/metrics/rl_comparison.csv>) |
| Ablation numerical results | [Ablation CSV](<D:/insurence seller/insurance_sales_agent/results/metrics/ablations.csv>) |
| Full recorded report | [Research report](<D:/insurence seller/insurance_sales_agent/results/reports/report.md>) |
| Actual training curve points | [Seed-42 training history](<D:/insurence seller/insurance_sales_agent/experiments/ppo_seed_42/run.json>) |
| Current feature status | [Specification status](<D:/insurence seller/insurance_sales_agent/docs/specification_status.md>) |
| Demo commands | [README](<D:/insurence seller/insurance_sales_agent/README.md>) |

### Method References

- John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford and Oleg Klimov. *Proximal Policy Optimization Algorithms*. 2017. [Paper](https://arxiv.org/abs/1707.06347).
- John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan and Pieter Abbeel. *High-Dimensional Continuous Control Using Generalized Advantage Estimation*. 2015 preprint, ICLR 2016. [Paper](https://arxiv.org/abs/1506.02438).
- Farama Foundation. [Gymnasium environment API](https://gymnasium.farama.org/api/env/).
- Ollama. [Chat API](https://docs.ollama.com/api/chat).

The five selected negotiation papers appear with their full titles and primary-source links in the literature survey section. Their role is to provide methodological context. All project-performance figures in these notes come from the saved local experiment outputs.
