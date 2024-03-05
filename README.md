# EcoSort: Your Waste Sorting Companion 🌱♻️

Managing waste effectively is crucial for the well-being of our environment and communities. EcoSort offers an innovative solution to tackle waste management challenges and promote sustainable practices.

## What EcoSort Does

- **Trash Sorting Education**: Engage in entertaining mini-games to learn about proper waste sorting techniques while having fun and staying motivated.
- **Image Recognition**: Utilize our user-friendly web application to instantly identify recyclable items by uploading their images.
- **Real-time Assistance**: Access a chatbot function for on-demand information about trash management techniques.

## Why Choose EcoSort?

- **Environmental Impact**: Contribute to environmental sustainability by adopting recycling and waste reduction practices.
- **Community Engagement**: Make waste sorting fun and rewarding for individuals and communities with interactive learning tools, including engaging mini-games.
- **Technological Innovation**: Benefit from cutting-edge image recognition technology to simplify waste sorting.

## Key Features

1. **Interactive Learning**: Engage in entertaining mini-games to learn efficient garbage sorting methods while enjoying the process.
2. **Image Recognition**: Upload item images to identify recyclable objects rapidly and precisely.
3. **Real-time Assistance**: Utilize the chatbot function for on-demand information about trash management techniques.

## How We Are Building It?

| Component          | Technology Stack                            |
|-------------------|---------------------------------------------|
| **Frontend**      | React, JavaScript, CSS                     |
| **Backend**       | Python, Django, PyTorch, Transformers      |
| **Image Recognition** | CNN (Convolutional Neural Network)      |
| **Chatbot**       | Natural Language Processing (NLP)          |
| **Mini-Game**     | Unity                                       |
## Getting Started
1. **Frontend Setup**: 
   - Navigate to the frontend directory and run it on live server to start the frontend server.

2. **Backend Setup**:
   - Navigate to the backend directory and  set up the transformer and move chatbot.py in huggingface_interface and run `python chatbot.py` to start the Flask server to run the chatbot.
3. **Transformer Setup**:  
   - Please Follow the following instructions for cloning the transformer
   ```bash
   # Clone the github repository and navigate to the project directory.
     git clone https://github.com/AI4Bharat/IndicTrans2
     cd IndicTrans2
     # Install all the dependencies and requirements associated with the project.
     source install.sh
   ```
   - Inside IndicTrans2 clone the tokeniser using the instructions below  
   ```bash 
   git clone https://github.com/VarunGumma/IndicTransTokenizer
   cd IndicTransTokenizer
   pip install --editable ./
   ```
   - Now navigate to example.py in huggingface_interface in IndicTrans2 and please paste this code instead of the existing one 
   ```python
      import torch
      from transformers import AutoModelForSeq2SeqLM
      from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer

      tokenizer = IndicTransTokenizer(direction="en-indic")
      ip = IndicProcessor(inference=True)
      model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-en-indic-dist-200M", trust_remote_code=True)

      sentences = [
          "This is a test sentence.",
          "This is another longer different test sentence.",
          "Please send an SMS to 9876543210 and an email on newemail123@xyz.com by 15th October, 2023.",
      ]

      batch = ip.preprocess_batch(sentences, src_lang="eng_Latn", tgt_lang="hin_Deva")
      batch = tokenizer(batch, src=True, return_tensors="pt")

      with torch.inference_mode():
          outputs = model.generate(**batch, num_beams=5, num_return_sequences=1, max_length=256)

      outputs = tokenizer.batch_decode(outputs, src=False)
      outputs = ip.postprocess_batch(outputs, lang="hin_Deva")
      print(outputs)
    
    ```

Note: We recommend creating a virtual environment with python>=3.7.

## Challenges Faced

1. **Limited Data Availability**: Dealing with diverse data sources required careful consideration to ensure accuracy.
2. **User Engagement**: Designing interactive and engaging tools, including mini-games, was crucial to encourage participation.

## Accomplishments to Be Proud Of

| Milestone                 | Description                                                 |
|---------------------------|-------------------------------------------------------------|
| Engaging User Interface  | Designed a visually appealing interface for enhanced user experience. |
| Efficient Image Recognition | Developed a reliable image recognition system for seamless waste sorting. |
| Real-time Assistance      | Implemented a chatbot function to provide instant help and guidance. |

## Lessons Learned

- **User-Centric Design**: Prioritizing user experience highlighted the importance of intuitive design and functionality.
- **Technological Innovation**: Leveraging advanced technologies like PyTorch, Transformers, and Unity expanded our capabilities.
- **Community Impact**: Engaging communities through education and technology, including mini-games, fosters a culture of environmental responsibility.

## What's Next for EcoSort?

| Next Steps                    | Description                                                 |
|-------------------------------|-------------------------------------------------------------|
| Expansion of Educational Resources | Develop additional educational content and games to further engage users. |
| Integration with Local Communities | Collaborate with local authorities and organizations to promote EcoSort adoption. |
| Continuous Improvement       | Gather user feedback to refine features and enhance usability. |
