// BasicSyntax.go
package main

import "fmt"

// Main function: Entry point of the program
func main() {
	// Variable declarations
	var message string = "Hello, Go!"
	var number int = 42
	isGoFun := true // Short variable declaration

	// Print statements
	fmt.Println(message)
	fmt.Printf("The number is: %d\n", number)
	fmt.Printf("Is Go fun? %t\n", isGoFun)

	// Conditional statement
	if isGoFun {
		fmt.Println("Yes, Go is fun!")
	} else {
		fmt.Println("No, Go is not fun.")
	}
}
