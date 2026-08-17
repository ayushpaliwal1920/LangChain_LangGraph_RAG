from langchain.text_splitter import RecursiveCharacterTextSplitter , MarkdownHeaderTextSplitter

text = """
In science, "space waves" typically refer to either Space Wave Propagation (used in radio and satellite communication) or Gravitational Waves (the literal ripples in the fabric of spacetime). Both concepts are fundamental to modern physics and telecommunications.1. Gravitational Waves (Astrophysics & Cosmology)Predicted by Albert Einstein in 1916, gravitational waves are invisible ripples in the fabric of spacetime caused by some of the most violent and energetic processes in the Universe.How they work: When massive objects like pairs of black holes or neutron stars orbit each other, they warp spacetime and emit ripples that travel at the speed of light. As they pass, they squeeze and stretch everything in their path.Detection: Detected on Earth using massive laser interferometers like LIGO, which measure unimaginably tiny changes (thousands of billions of times smaller than a proton) to confirm cosmic events."""

# Initialize the splitter :

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 50,
    chunk_overlap = 10,
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks)