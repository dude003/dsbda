object LargerOfTwo {
  def main(args: Array[String]): Unit = {
      println("Enter the first number:")
      val num1 = scala.io.StdIn.readDouble()
      println("Enter the second number:")
      val num2 = scala.io.StdIn.readDouble()
      if(num1 == num2){
        println(s"Both Numbers are same: $num2")
      } else{
          val largerNumber = if (num1 > num2) num1 else num2
          println(s"The larger number is: $largerNumber")
      }
  }
}

object NumberCheck {
  def main(args: Array[String]): Unit = {
      println("Enter a number:")
      val num = scala.io.StdIn.readDouble()
      if (num > 0) {
      println("The number is positive.")
    } else if (num < 0) {
      println("The number is negative.")
    } else {
      println("The number is zero.")
    }
  }
}