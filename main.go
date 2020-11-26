package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"strconv"

	"github.com/gorilla/mux"
)

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
