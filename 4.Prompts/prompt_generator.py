from langchain_core.prompts import PromptTemplate

# template :


template = PromptTemplate(
     template="""
You are an expert AI Research Assistant.

Explain the research paper:
"{paper_input}"

Explanation Style:
{style_input}

Explanation Length:
{length_input}

Your explanation should:
- Be well structured
- Use simple and clear language where appropriate
- Include key concepts and architecture
- Mention important innovations
- Explain real-world applications
- Include examples if helpful

Start the explanation now.
""",
input_variables=['paper_input' , 'style_input' , 'length_input'],
validate_template=True
)

template.save("template.json")