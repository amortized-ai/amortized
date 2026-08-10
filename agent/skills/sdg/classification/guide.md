# Classification -- SDG Workflow

Generate labeled training data for ticket classifiers, intent routers,
sentiment analyzers, or content moderators.

## Workflow

Ask the user these questions one at a time using `present_options`.

### Step 1 -- Domain

What kind of content will this classifier handle?

Present options:
- Software/technical support -- Bug reports, feature requests, troubleshooting
- Billing & payments -- Invoices, refunds, subscription issues
- Customer service -- Account access, onboarding, general inquiries
- E-commerce -- Orders, shipping, returns, product questions

### Step 2 -- Categories

Based on the domain, suggest specific category labels (3-6). Let the
user customize or define their own.

For customer support, present:
- Standard (4 categories): Billing, Technical, Account, General Inquiry
- Detailed (6 categories): Billing, Technical, Account, Shipping, Returns, Product Questions
- Custom: user defines their own labels

Adapt suggestions to the chosen domain.

### Step 3 -- Urgency levels

Should the classifier also assign urgency?
- Yes, 3 levels -- Low, Medium, High
- Yes, 4 levels -- Low, Medium, High, Critical
- No urgency -- classify by category only

### Step 4 -- Teacher model

Call `list_models` to get models from the AI Gateway. Present ONLY
those models. If none returned, direct user to Settings -> AI Gateway.

### Step 5 -- Sample count

Scale based on category count and urgency levels.
Let N = number of categories x urgency levels (or just categories
if no urgency).

Present tiers:
1. N x 50 samples -- Basic coverage
2. N x 100 samples -- Good diversity, recommended
3. N x 150 samples -- Best quality

### Step 6 -- Distribution

Should categories be balanced or weighted? Default: roughly balanced
unless the real-world distribution is known.

## Reference Payload

Use this as the base for `create_sdg_job()`. This example shows a
customer support ticket classifier with 4 categories and 3 urgency
levels. Adapt categories, prompts, and sample count to the user's
domain.

```json
{
  "num_records": 400,
  "topic": "customer support classification",
  "columns": [
    {
      "column_type": "sampler",
      "name": "category",
      "sampler_type": "category",
      "params": {
        "values": [
          "Billing - Invoice disputes, payment failures, refund requests, subscription changes",
          "Technical - Bug reports, feature requests, integration issues, API errors",
          "Account - Password resets, access issues, profile updates, account closures",
          "General Inquiry - Product questions, pricing, onboarding help, feedback"
        ],
        "weights": [0.25, 0.30, 0.25, 0.20]
      }
    },
    {
      "column_type": "sampler",
      "name": "urgency",
      "sampler_type": "category",
      "params": {
        "values": [
          "Low - Informational or non-blocking issue",
          "Medium - Affects workflow but has a workaround",
          "High - Blocking issue requiring prompt attention"
        ],
        "weights": [0.40, 0.35, 0.25]
      }
    },
    {
      "column_type": "llm-text",
      "name": "ticket_text",
      "model_alias": "text",
      "system_prompt": "You generate realistic customer support tickets. Write a natural, varied message as a customer would write it -- including typos, informal language, and varying levels of detail. The ticket must clearly belong to the specified category and urgency level. Output ONLY the ticket text.",
      "prompt": "Category: {{ category }}\nUrgency: {{ urgency }}"
    },
    {
      "column_type": "llm-text",
      "name": "label",
      "model_alias": "text",
      "system_prompt": "You are a support ticket classifier. Given a customer message, output ONLY the category label -- one of the predefined categories. No explanation, no formatting, just the label.",
      "prompt": "Ticket:\n{{ ticket_text }}\n\nClassify into one of: Billing, Technical, Account, General Inquiry"
    }
  ],
  "model_configs": [
    {
      "alias": "text",
      "model": "<from-list_models>",
      "provider": "gateway",
      "skip_health_check": true,
      "inference_parameters": {
        "temperature": 0.7,
        "max_parallel_requests": 32
      }
    }
  ],
  "processors": [
    {
      "processor_type": "schema_transform",
      "name": "sft_format",
      "template": {
        "messages": [
          {"role": "system", "content": "You are a customer support classifier. Categorize the incoming message into one of: Billing, Technical, Account, General Inquiry. Respond with only the category label."},
          {"role": "user", "content": "{{ ticket_text }}"},
          {"role": "assistant", "content": "{{ label }}"}
        ]
      }
    }
  ]
}
```

### Adapting the Payload

- **`columns[0].params.values`**: match the user's categories from Step 2
- **`columns[1]`**: include urgency sampler only if user chose urgency in Step 3; remove entirely otherwise
- **`columns[2].prompt`**: include `\nUrgency: {{ urgency }}` only if urgency sampler exists
- **`columns[3].prompt`**: list the actual category labels the user chose
- **`model_configs[0].model`**: the model chosen in Step 4
- **`processors[0].template.messages[0].content`**: list the actual category labels
- **`num_records`**: from Step 5 calculation
- **`columns[*].params.weights`**: match the user's distribution choice from Step 6

## After SDG -- Training

Recommend OSFT training. Read `skills/training/knowledge-ingestion/osft/guide.md`.
Chain via `parent_job_id`.
