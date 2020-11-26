# tf-concurrente-backend
Backend TF Concurrente UPC

#Move to source folder
cd src

#Install dependencies
go get -u github.com/gorilla/mux
go get github.com/githubnemo/CompileDaemon

#Execute with hotreaload
CompileDaemon -command="tf-concurrente-backend.exe"