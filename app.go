package main

// Example implementation

func main() {

	// Supply the Caffe2 model file location
	model, err := New("basic-model.c2")
	if err != nil {
		panic(err)
	}

	model2, err := New("basic-model.c2")
	if err != nil {
		panic(err)
	}

	{
		// Predict the sentence with that Caffe2 model
		sentence := "Sentence to predict"
		err = model.Predict(sentence)
		if err != nil {
			panic(err)
		}
	}

	{
		// Predict the sentence with that Caffe2 model
		sentence := "Sentence to predict"
		err = model.Predict(sentence)
		if err != nil {
			panic(err)
		}
	}

	{
		// Predict the sentence with that Caffe2 model
		sentence := "Sentence to predict"
		err = model.Predict(sentence)
		if err != nil {
			panic(err)
		}
	}

	{
		// Predict the sentence with that Caffe2 model
		sentence2 := "Second sentence to predict"
		err = model2.Predict(sentence2)
		if err != nil {
			panic(err)
		}
	}

	{
		// Predict the sentence with that Caffe2 model
		sentence2 := "Second sentence to predict"
		err = model2.Predict(sentence2)
		if err != nil {
			panic(err)
		}
	}
}
