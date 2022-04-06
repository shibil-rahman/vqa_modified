from simpletransformers.language_representation import RepresentationModel
def text_feature_extraction(sentences):

    model = RepresentationModel(
            model_type="bert",
            model_name="bert-base-uncased",
            use_cuda=False
        )

    #The RepresentationModel class is used for generating (contextual) word or sentence embeddings from a list of text sentences.
    #It must specify a model_type and a model_name. model_type should be one of the model types, currently supported: bert, roberta, gpt2.

    sentence_vectors = model.encode_sentences(sentences, combine_strategy="mean") 
    #The encode_sentences() method is used to create word embeddings with the model. Generates list of contextual word or sentence embeddings using the model passed to class constructor.
    #Prameters: text_list - list of text sentences, combine_strategy - strategy for combining word vectors-- supported values: None, “mean”, “concat”, batch_size - size of batches of sentences feeded to the model.
    
    print("sentence vector shape",sentence_vectors.shape)
    return sentence_vectors
