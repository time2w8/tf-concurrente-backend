package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"net/http"
	"strconv"
	"time"

	"github.com/gorilla/mux"
)

// SOBRE LA RED NEURONAL

// estructura de la red neuronal (perceptrón multicapa)
type NeuralNetwork struct {
	hiddenWeights       [][]float64
	outputWeights       [][]float64
	hiddenBias          []float64
	outputBias          []float64
	epochErrors         []float64
	modelError          float64
	learningRate        float64
	epochs              int
	lengthTrainingSet   int
	inputNeuronsAmount  int
	hiddenNeuronsAmount int
	outputNeuronsAmount int
}

// función para inicializar la red neuronal
func (n *NeuralNetwork) initializeNetwork() {
	rand.Seed(time.Now().UnixNano())

	// inicialización de bias
	n.hiddenBias = make([]float64, n.hiddenNeuronsAmount)
	n.outputBias = make([]float64, n.outputNeuronsAmount)
	for i := 0; i < n.hiddenNeuronsAmount; i++ {
		n.hiddenBias[i] = rand.Float64()
	}
	for i := 0; i < n.outputNeuronsAmount; i++ {
		n.outputBias[i] = rand.Float64()
	}

	// inicialización de pesos
	n.hiddenWeights = make([][]float64, n.hiddenNeuronsAmount)
	n.outputWeights = make([][]float64, n.outputNeuronsAmount)
	for i := 0; i < n.hiddenNeuronsAmount; i++ {
		n.hiddenWeights[i] = make([]float64, n.inputNeuronsAmount)
		for j := 0; j < n.inputNeuronsAmount; j++ {
			n.hiddenWeights[i][j] = rand.Float64()
		}
	}
	for i := 0; i < n.outputNeuronsAmount; i++ {
		n.outputWeights[i] = make([]float64, n.hiddenNeuronsAmount)
		for j := 0; j < n.hiddenNeuronsAmount; j++ {
			n.outputWeights[i][j] = rand.Float64()
		}
	}

	// creación de vector para almacenar los errores de cada época
	n.epochErrors = make([]float64, n.epochs)
}

// FUNCIONES CON MATRICES

// obtiene solo los inputs del dataset de entrenamiento
func getInputSetOnly(trainingSet [][]float64) [][]float64 {
	inputs := make([][]float64, len(trainingSet))
	numberOfExpectedOutputs := 1

	for i := 0; i < len(trainingSet); i++ {
		inputs[i] = make([]float64, len(trainingSet[i])-numberOfExpectedOutputs)
		for j := 0; j < len(trainingSet[i])-numberOfExpectedOutputs; j++ {
			inputs[i][j] = trainingSet[i][j]
		}
	}
	return inputs
}

// obtiene solo los valores esperados del dataset de entrenamiento
func getExpectedValuesOnly(trainingSet [][]float64) [][]float64 {
	expected := make([][]float64, len(trainingSet))
	numberOfExpectedOutputs := 1

	for i := 0; i < len(trainingSet); i++ {
		expected[i] = make([]float64, numberOfExpectedOutputs)
		for j := 0; j < 1; j++ {
			expected[i][j] = trainingSet[i][len(trainingSet[i])-numberOfExpectedOutputs]
		}
	}
	return expected
}

// obtiene la matriz transpuesta de una matriz
func getMatTransposed(m1 [][]float64) [][]float64 {
	transposed := make([][]float64, len(m1[0]))

	for i := 0; i < len(m1[0]); i++ {
		transposed[i] = make([]float64, len(m1))
		for j := 0; j < len(m1); j++ {
			transposed[i][j] = m1[j][i]
		}
	}
	return transposed
}

// calcula la multipicación escalar entre dos matrices
func matDotProduct(m1, m2 [][]float64) [][]float64 {
	product := make([][]float64, len(m1))

	for i := 0; i < len(m1); i++ {
		product[i] = make([]float64, len(m2))
		for j := 0; j < len(m2); j++ {
			product[i][j] = vecDotProduct(m1[i], m2[j])
		}
	}
	return product
}

// calcula la adición entre dos vectores
func matAdd(m1, m2 [][]float64) [][]float64 {
	add := make([][]float64, len(m1))

	for i := 0; i < len(m1); i++ {
		add[i] = make([]float64, len(m1[i]))
		for j := 0; j < len(m1[i]); j++ {
			add[i][j] = m1[i][j] + m2[i][j]
		}
	}
	return add
}

// calcula la multiplicación de una matriz y un escalar
func scalarMatMulitply(m1 [][]float64, s float64) [][]float64 {
	product := make([][]float64, len(m1))

	for i := 0; i < len(m1); i++ {
		product[i] = make([]float64, len(m1[i]))
		for j := 0; j < len(m1[i]); j++ {
			product[i][j] = m1[i][j] * s
		}
	}
	return product
}

// obtiene el error diferencial entre el valor esperado y el valor obtenido
func getDifError(y, d [][]float64) [][]float64 {
	dif := make([][]float64, len(y))

	for i := 0; i < len(y); i++ {
		dif[i] = make([]float64, 1)
		dif[i][0] = d[i][0] - y[i][0]
	}
	return dif
}

// convierte la matriz de string a float64
func matConvertToFloat64(m1 [][]string) [][]float64 {
	converted := make([][]float64, len(m1))
	for i := 0; i < len(m1); i++ {
		converted[i] = make([]float64, len(m1[i]))
		for j := 0; j < len(m1[i]); j++ {
			converted[i][j], _ = strconv.ParseFloat(m1[i][j], 64)
		}
	}
	return converted
}

// FUNCIONES CON VECTORES

// calcula la multiplicación escalar entre dos vectores
func vecDotProduct(v1, v2 []float64) float64 {
	product := 0.0

	for i := 0; i < len(v1); i++ {
		product += v1[i] * v2[i]
	}
	return product
}

// calcula la adición entre dos vectores
func vecAdd(v1, v2 []float64) []float64 {
	add := make([]float64, len(v1))

	for i := 0; i < len(v1); i++ {
		add[i] = v1[i] + v2[i]
	}
	return add
}

// calcula la multiplicación de un vector y un escalar
func scalarVecMutliply(v1 []float64, s float64) []float64 {
	product := make([]float64, len(v1))

	for i := 0; i < len(v1); i++ {
		product[i] = v1[i] * s
	}
	return product
}

// multiplica los valores de v1 con sus contrapartes en v2
func vecMutliply(v1, v2 []float64) []float64 {
	product := make([]float64, len(v1))

	for i := 0; i < len(v1); i++ {
		product[i] = v1[i] * v2[i]
	}
	return product
}

// reduce una matriz a un vector
func matReduce(m1 [][]float64) []float64 {
	reduced := make([]float64, len(m1[0]))

	for i := 0; i < len(m1[0]); i++ {
		reduced[i] = 0
		for j := 0; j < len(m1); j++ {
			reduced[i] += m1[j][i]
		}
	}
	return reduced
}

// función sigmoide
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// derivada de la función sigmoide
func sigmoidDerivative(y float64) float64 {
	return y * (1 * y)
}

// propaga hacia adelante los valores de entrada del dataset
func (n *NeuralNetwork) frontPropagation(chFrontPropagation chan string, trainingSet [][]float64, numberOfSet int) ([][]float64, [][]float64) {
	// propagación de la capa de entrada a la capa oculta
	hiddenPropagation := make([][]float64, len(trainingSet))
	for i := 0; i < len(trainingSet); i++ {
		hiddenPropagation[i] = make([]float64, n.hiddenNeuronsAmount)
		for j := 0; j < n.hiddenNeuronsAmount; j++ {
			hiddenPropagation[i][j] = sigmoid(vecDotProduct(trainingSet[i][:n.inputNeuronsAmount], n.hiddenWeights[j]) + n.hiddenBias[j])
		}
	}

	// propagación de la capa oculta a la capa de salida
	outputPropagation := make([][]float64, len(trainingSet))
	for i := 0; i < len(trainingSet); i++ {
		outputPropagation[i] = make([]float64, n.outputNeuronsAmount)
		for j := 0; j < n.outputNeuronsAmount; j++ {
			outputPropagation[i][j] = sigmoid(vecDotProduct(hiddenPropagation[i], n.hiddenWeights[j]) + n.outputBias[j])
		}
	}

	// se avisa al canal que ya terminó la propagación en esta función
	chFrontPropagation <- fmt.Sprintf("Propagated set %d", numberOfSet)

	// devolución los valores finales de la capa intermedia y oculta, ya que se usaran en la propagación hacia atrás
	return hiddenPropagation, outputPropagation
}

// función de la propagación hacia atrás
func (n *NeuralNetwork) backPropagation(hiddenPropagation, outputPropagation, trainingSet [][]float64, epoch int) {
	// cálculo del error de la capa de salida
	outputPredictedError := getDifError(outputPropagation, getExpectedValuesOnly(trainingSet))

	// cálculo de la derivada sigmoidal de los valores finales de la capa de salida
	derivativeOutputPropagation := make([][]float64, len(outputPropagation))
	for i := 0; i < len(outputPropagation); i++ {
		derivativeOutputPropagation[i] = make([]float64, len(outputPropagation[i]))
		for j := 0; j < len(outputPropagation[i]); j++ {
			derivativeOutputPropagation[i][j] = sigmoidDerivative(outputPropagation[i][j])
		}
	}

	// cálculo del delta de la capa de salida
	outputPredictedDelta := make([][]float64, len(outputPredictedError))
	for i := 0; i < len(outputPredictedError); i++ {
		outputPredictedDelta[i] = vecMutliply(derivativeOutputPropagation[i], outputPredictedError[i])
	}

	// cálculo del error de la capa intermedia
	hiddenPredictedError := matDotProduct(outputPredictedDelta, getMatTransposed(n.outputWeights))

	// cálculo de la derivada sigmoidal de los valores finales de la capa intermedia
	derivativeHiddenPropagation := make([][]float64, len(hiddenPropagation))
	for i := 0; i < len(hiddenPropagation); i++ {
		derivativeHiddenPropagation[i] = make([]float64, len(hiddenPropagation[i]))
		for j := 0; j < len(hiddenPropagation[i]); j++ {
			derivativeHiddenPropagation[i][j] = sigmoidDerivative(hiddenPropagation[i][j])
		}
	}

	// cálculo del delta de la capa intermedia
	hiddenPredictedDelta := make([][]float64, len(hiddenPredictedError))
	for i := 0; i < len(outputPredictedError); i++ {
		hiddenPredictedDelta[i] = vecMutliply(derivativeHiddenPropagation[i], hiddenPredictedError[i])
	}

	// actualización de los pesos y bias
	n.outputWeights = matAdd(n.outputWeights, scalarMatMulitply(getMatTransposed(matDotProduct(getMatTransposed(hiddenPropagation), getMatTransposed(outputPredictedDelta))), n.learningRate))
	n.outputBias = vecAdd(n.outputBias, scalarVecMutliply(matReduce(outputPredictedDelta), n.learningRate))
	n.hiddenWeights = matAdd(n.hiddenWeights, scalarMatMulitply(getMatTransposed(matDotProduct(getMatTransposed(getInputSetOnly(trainingSet)), getMatTransposed(hiddenPredictedDelta))), n.learningRate))
	n.hiddenBias = vecAdd(n.hiddenBias, scalarVecMutliply(matReduce(hiddenPredictedDelta), n.learningRate))

	// cálculo y almacén del error de la época
	reducedPredictedError := matReduce(outputPredictedError)[0] / float64(len(trainingSet))
	n.epochErrors[epoch] = reducedPredictedError * reducedPredictedError
}

// FUNCIONES CON ARCHIVOS CSV

// obtiene y lee el archivo csv de la url especificada
func readCSVFromUrl(url string) ([][]string, error) {
	resp, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	fmt.Println(resp.Body)
	defer resp.Body.Close()
	reader := csv.NewReader(resp.Body)
	reader.Comma = ','
	data, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	return data, nil
}

type user struct {
	ID       int    `json:ID`
	Name     string `json:Name`
	LastName string `json:LastName`
	Password string `json:Password`
}

type userList []user

var users = userList{
	{
		ID:       1,
		Name:     "Emanuel",
		LastName: "Gonzales",
		Password: "12345",
	},
}

// --GET-- {API_URL}/users
func getUsers(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(users)
}

// --POST-- {API_URL}/users
func createUser(w http.ResponseWriter, r *http.Request) {
	var myUser user
	reqBody, err := ioutil.ReadAll(r.Body)

	if err != nil {
		fmt.Fprintf(w, "Inserte un usuario valido.")
	}

	json.Unmarshal(reqBody, &myUser)

	myUser.ID = len(users) + 1
	users = append(users, myUser)

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(myUser)
}

// --GET-- {API_URL}/users/{id}
func getUser(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	userID, err := strconv.Atoi(vars["id"])

	if err != nil {
		fmt.Fprintf(w, "Invalid ID.")
	}

	for _, user := range users {
		if user.ID == userID {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(user)
		}
	}
}

// --DELETE-- {API_URL}/users/{id}
func deleteUser(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	userID, err := strconv.Atoi(vars["id"])

	if err != nil {
		fmt.Fprintf(w, "Id no válido")
	}

	for index, user := range users {
		if user.ID == userID {
			users = append(users[:index], users[index+1:]...)
			fmt.Fprintf(w, "El usuario con id %v ha sido removido.", user.ID)
		}
	}
}

// --PUT-- {API_URL}/users/{id}
func updateUser(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	userID, err := strconv.Atoi(vars["id"])
	var updatedUser user

	if err != nil {
		fmt.Fprintf(w, "Id no válido")
	}

	reqBody, err := ioutil.ReadAll(r.Body)
	if err != nil {
		fmt.Fprintf(w, "Inserte un usuario valido.")
	}

	json.Unmarshal(reqBody, &updatedUser)

	for index, user := range users {
		if user.ID == userID {
			users = append(users[:index], users[index+1:]...)
			updatedUser.ID = userID
			users = append(users, updatedUser)
			fmt.Fprintf(w, "The task with ID %v has been updated", user.ID)
		}
	}
}

func main() {
	// inicialización eel tiempo (para medir cuánto se demora en ejecutarse)
	start := time.Now()

	// obtenemos el datset de entrenamiento desde el repositorio online
	url := "https://raw.githubusercontent.com/AdrianCAmes/Go_Parallel_Backpropagation/main/iris_dataset.csv"
	data, _ := readCSVFromUrl(url)
	trainingSet := matConvertToFloat64(data)

	// inicialización de la red neuronal
	network := NeuralNetwork{
		lengthTrainingSet:   len(trainingSet),
		inputNeuronsAmount:  4,
		hiddenNeuronsAmount: 2,
		outputNeuronsAmount: 1,
		epochs:              1000,
		learningRate:        0.3,
	}
	network.initializeNetwork()

	// división del dataset en 4 secciones de entrenamiento
	lengthTrainingSet := network.lengthTrainingSet
	firstTrainingSection := trainingSet[:lengthTrainingSet/4]
	secondTrainingSection := trainingSet[lengthTrainingSet/4 : 2*lengthTrainingSet/4]
	thirdTrainingSection := trainingSet[2*lengthTrainingSet/4 : 3*lengthTrainingSet/4]
	fourthTrainingSection := trainingSet[3*lengthTrainingSet/4:]

	// inicialización y cierre anticipado del canal por el cual se compartirá información de la propagaciónh hacia adelante
	chFrontPropagation := make(chan string)
	defer close(chFrontPropagation)

	// ciclo for por cada época
	for i := 0; i < network.epochs; i++ {
		fmt.Printf("Epoch %d \n", i)

		// ejecución paralela de la propagación hacia adelante de cada sección de entrenamiento
		finishedSets := 0
		var output1, hidden1, output2, hidden2, output3, hidden3, output4, hidden4 [][]float64
		go func() { hidden1, output1 = network.frontPropagation(chFrontPropagation, firstTrainingSection, 1) }()
		go func() { hidden2, output2 = network.frontPropagation(chFrontPropagation, secondTrainingSection, 2) }()
		go func() { hidden3, output3 = network.frontPropagation(chFrontPropagation, thirdTrainingSection, 3) }()
		go func() { hidden4, output4 = network.frontPropagation(chFrontPropagation, fourthTrainingSection, 4) }()
		for {
			msg := <-chFrontPropagation
			fmt.Printf("Msg: %s \n", msg)
			if msg[:10] == "Propagated" {
				finishedSets += 1
			}
			// si todas las secciones se han propagado hacia adelante, se rompe el ciclo
			if finishedSets == 4 {
				time.Sleep(360000 * time.Nanosecond)
				break
			}
		}
		// combinación de los resultados de cada sección de aprendizaje
		propagatedGeneralOutput := make([][]float64, lengthTrainingSet)
		propagatedGeneralHidden := make([][]float64, lengthTrainingSet)
		// -- combinación de la primera sección
		indexHelper := 0
		for j := 0; j < len(output1); j++ {
			propagatedGeneralOutput[j+indexHelper] = output1[j]
			propagatedGeneralHidden[j+indexHelper] = hidden1[j]
		}
		// -- combinación de la segunda sección
		indexHelper = indexHelper + len(output1)
		for j := 0; j < len(output2); j++ {
			propagatedGeneralOutput[j+indexHelper] = output2[j]
			propagatedGeneralHidden[j+indexHelper] = hidden2[j]
		}
		// -- combinación de la tercera sección
		indexHelper = indexHelper + len(output2)
		for j := 0; j < len(output3); j++ {
			propagatedGeneralOutput[j+indexHelper] = output3[j]
			propagatedGeneralHidden[j+indexHelper] = hidden3[j]
		}
		// -- combinación de la cuarta sección
		indexHelper = indexHelper + len(output3)
		for j := 0; j < len(output4); j++ {
			propagatedGeneralOutput[j+indexHelper] = output4[j]
			propagatedGeneralHidden[j+indexHelper] = hidden4[j]
		}

		// propagación hacia atrás y aprendizaje de la red neuronal
		network.backPropagation(propagatedGeneralHidden, propagatedGeneralOutput, trainingSet, i)
		fmt.Printf("Learned in epoch %d\n\n", i)
	}

	// cálculo del error general del modelo
	for i := 0; i < len(network.epochErrors); i++ {
		network.modelError += network.epochErrors[i] / float64(network.epochs)
	}

	// cálculo del tiempo transcurrido
	elapsed := time.Since(start)

	// visualización de métricas
	fmt.Printf("\n---------------------------\n")
	fmt.Printf("METRICS")
	fmt.Printf("\n---------------------------\n")
	fmt.Printf("Model error: %f\n", network.modelError)
	fmt.Printf("Execution took %s", elapsed)
	router := mux.NewRouter().StrictSlash(true)

	//USER ROUTES
	router.HandleFunc("/users", getUsers).Methods("GET")
	router.HandleFunc("/users", createUser).Methods("POST")
	router.HandleFunc("/users/{id}", getUser).Methods("GET")
	router.HandleFunc("/users/{id}", deleteUser).Methods("DELETE")
	router.HandleFunc("/users/{id}", updateUser).Methods("PUT")
	//-----------

	log.Fatal(http.ListenAndServe(":3000", router))
}
