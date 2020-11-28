package main

// importar paquetes necesarios
import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"net"
	"net/http"
	"os"
	"strconv"
	"time"
	//"reflect"
	"github.com/gorilla/mux"
	"io/ioutil"
	"log"
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
	inputNeuronsAmount  int
	hiddenNeuronsAmount int
	outputNeuronsAmount int
}

// configuración inicial del network
var network = NeuralNetwork{
	inputNeuronsAmount:  4,
	hiddenNeuronsAmount: 2,
	outputNeuronsAmount: 1,
	epochs:              5,
	learningRate:        0.3,
}
var chMsg chan Msg

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

// inicialización de la red neuronal
var network = NeuralNetwork{
	inputNeuronsAmount:  4,
	hiddenNeuronsAmount: 2,
	outputNeuronsAmount: 1,
	epochs:              1000,
	learningRate:        0.3,
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
	return y * (1 - y)
}

// propagar hacia delante para predecir
func (n *NeuralNetwork) predict(predictData [][]float64) float64 {
	// propagación de la capa de entrada a la capa oculta
	hiddenPropagation := make([][]float64, len(predictData))
	for i := 0; i < len(predictData); i++ {
		hiddenPropagation[i] = make([]float64, n.hiddenNeuronsAmount)
		for j := 0; j < n.hiddenNeuronsAmount; j++ {
			hiddenPropagation[i][j] = sigmoid(vecDotProduct(predictData[i][:n.inputNeuronsAmount], n.hiddenWeights[j]) + n.hiddenBias[j])
		}
	}

	// propagación de la capa oculta a la capa de salida
	outputPropagation := make([][]float64, len(predictData))
	for i := 0; i < len(predictData); i++ {
		outputPropagation[i] = make([]float64, n.outputNeuronsAmount)
		for j := 0; j < n.outputNeuronsAmount; j++ {
			outputPropagation[i][j] = sigmoid(vecDotProduct(hiddenPropagation[i], n.outputWeights[j]) + n.outputBias[j])
		}
	}

	// devolución los valores finales de la capa intermedia y oculta, ya que se usaran en la propagación hacia atrás
	return math.Round(outputPropagation[0][0])
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
			outputPropagation[i][j] = sigmoid(vecDotProduct(hiddenPropagation[i], n.outputWeights[j]) + n.outputBias[j])
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
func trainModel(trainUrl string) {
	// inicialización eel tiempo (para medir cuánto se demora en ejecutarse)
	start := time.Now()

	// obtenemos el datset de entrenamiento desde el repositorio online
	url := trainUrl
	data, _ := readCSVFromUrl(url)
	trainingSet := matConvertToFloat64(data)

	// división del dataset en 4 secciones de entrenamiento
	lengthTrainingSet := len(trainingSet)
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

	fmt.Printf("Model error: %f\n", network.modelError*10)
	fmt.Printf("Execution took %s\n", elapsed)
}


//----------------------------------------------------------------------------------

type Msg struct {
	Addr string `json:"addr"`
	Data string `json:"data"`
}

type PredictData struct {
	Education float64 `json:"Education"`
	Salary float64 `json:"Salary"`
	Loan float64 `json:"Loan"`
	History float64 `json:"History"`
}

type TrainData struct {
	Url string `json:"url"`
}

type Reciever struct {
	Addr string `json:"addr"`
	Data string `json:"data"`
	Education float64 `json:"Education"`
	Salary float64 `json:"Salary"`
	Loan float64 `json:"Loan"`
	History float64 `json:"History"`
	Url string `json:"url"`
}

func server(local string, remotes []string, chMsg chan Msg) {
	if ln, err := net.Listen("tcp", local); err == nil {
		defer ln.Close()
		fmt.Printf("Listening on %s\n", local)
		if conn, err := ln.Accept(); err == nil {
			fmt.Printf("Connection accepted from %s\n", conn.RemoteAddr())
			handle(conn, local, remotes, chMsg)
		}
	}
}

func handle(conn net.Conn, local string, remotes []string, chMsg chan Msg) {
	defer conn.Close()
	dec := json.NewDecoder(conn)
	var msg Msg
	var pd PredictData
	var rv Reciever
	if err := dec.Decode(&rv); err == nil  {
		fmt.Println(rv)
		if !(rv.Addr == "" && rv.Data == "") {
			msg.Addr = rv.Addr
			msg.Data = rv.Data
			fmt.Printf("Message: %v\n", msg)
			chMsg <- msg
		} else if !(rv.Education == 0 && rv.Salary == 0 && rv.Loan == 0 && rv.History == 0) {	
			pd.Education = rv.Education
			pd.Salary = rv.Salary
			pd.Loan = rv.Loan
			pd.History = rv.History
			predictHandler(chMsg, pd, local, remotes)
			fmt.Printf("Predicting data: %v\n", pd)
		} else if !(rv.Url == "") {
			trainModel(rv.Url)
			fmt.Printf("Training data: %v\n", rv.Url)
		}
	} 
}

func sendAll(option, local string, remotes []string) {
	for _, remote := range remotes {
		send(local, remote, option)
	}
}

func send(local, remote, option string) {
	if conn, err := net.Dial("tcp", remote); err == nil {
		enc := json.NewEncoder(conn)
		if err := enc.Encode(Msg{local, option}); err == nil {
			fmt.Printf("Sending %s to %s\n", option, remote)
		}
	}
}

func sendAllTrain(url, local string, remotes []string) {
	for _, remote := range remotes {
		sendTrain(local, remote, url)
	}
}

func sendTrain(local, remote, url string) {
	if conn, err := net.Dial("tcp", remote); err == nil {
		enc := json.NewEncoder(conn)
		if err := enc.Encode(TrainData{url}); err == nil {
			fmt.Printf("Sending %s to %s\n", url, remote)
		}
	}
}

func sendAllPredict(object PredictData, local string, remotes []string) {
	for _, remote := range remotes {
		sendPredict(local, remote, object)
	}
}

func sendPredict(local, remote string, object PredictData) {
	if conn, err := net.Dial("tcp", remote); err == nil {
		enc := json.NewEncoder(conn)
		if err := enc.Encode(object); err == nil {
			fmt.Printf("Sending %s to %s\n", object, remote)
		}
	}
}

func executeServer(chMsg chan Msg, local string, remotes []string) {
	go server(local, remotes, chMsg)
}

func predictHandler(chMsg chan Msg, predictData PredictData, local string, remotes []string) string {
	predictMatrix := make([][]float64, 1)
	predictMatrix[0] = make([]float64, network.inputNeuronsAmount)
	predictMatrix[0][0] = predictData.Education
	predictMatrix[0][1] = predictData.Salary
	predictMatrix[0][2] = predictData.Loan
	predictMatrix[0][3] = predictData.History
	data := "Riesgo"
	var riesgo, sin_riesgo int

	predictedValue := network.predict(predictMatrix)
	if predictedValue == 1 {
		data = "Seguro"
	}
	sendAll(data, local, remotes)
	if data == "Riesgo" {
		riesgo = 1
	} else {
		sin_riesgo = 1
	}
	/*for range remotes {
		fmt.Println("here")
		msg := <-chMsg
		fmt.Println("there")
		if msg.Data == "Riesgo" {
			riesgo++
		} else {
			sin_riesgo++
		}
	}*/
	if riesgo > sin_riesgo {
		fmt.Println("Tiene riesgo")
		return "Riesgo"
	} else {
		fmt.Println("Es seguro")
		return "Seguro"
	}
}

func trainNeuronalNetwork(w http.ResponseWriter, r *http.Request) {
	var td TrainData
	reqBody, err := ioutil.ReadAll(r.Body)
	if err != nil {
		fmt.Fprintf(w, "Inserte un cuerpo valido.")
	}
	json.Unmarshal(reqBody, &td)

	local := os.Args[1]
	remotes := os.Args[2:]
	
	sendAllTrain(td.Url, local, remotes)
	trainModel(td.Url)
	fmt.Fprintf(w, "Model error: %f\n", network.modelError)
	executeServer(chMsg, local, remotes)
}

func makePrediction(w http.ResponseWriter, r *http.Request) {
	var pd PredictData
	reqBody, err := ioutil.ReadAll(r.Body)
	if err != nil {
		fmt.Fprintf(w, "Inserte un cuerpo valido.")
	}
	json.Unmarshal(reqBody, &pd)

	local := os.Args[1]
	remotes := os.Args[2:]
	sendAllPredict(pd, local, remotes)
	predictedValue := predictHandler(chMsg, pd,local, remotes)
	fmt.Fprintf(w, "Prediction result: %f\n", predictedValue)
	executeServer(chMsg, local, remotes)
}

func main() {
	router := mux.NewRouter().StrictSlash(true)
	local := os.Args[1]
	remotes := os.Args[2:]
	chMsg = make(chan Msg)
	executeServer(chMsg, local, remotes)

	//RED NEURONAL ROUTES
	router.HandleFunc("/train", trainNeuronalNetwork).Methods("POST")
	router.HandleFunc("/prediction", makePrediction).Methods("POST")
	
	// inicialización de la red neuronal
	network.initializeNetwork()
	//-----------
	
	var str string
	port := ":3"
	for i := 0; i < 3; i++ {
		str = string(local[len(local)-(3-i)])
		num, err := strconv.ParseInt(str, 10, 64)
		if err != nil {
			fmt.Println("Bad Port")
		}
		port += strconv.FormatInt(num, 16)
	}
	fmt.Println("API on port", port)
	log.Fatal(http.ListenAndServe(port, router))

}
