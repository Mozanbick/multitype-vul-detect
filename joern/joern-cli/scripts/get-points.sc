/* get-points.scala

   This script collects suspicious vulnerable statements using nodes' properties, and returns the id of corresponding Cpg Elements.

   Input: A valid CPG
   Output: scala.List[Long]

   Running the Script
   ------------------
   Use joern command line tool
	 Run command: cpg.runScript("get-points.sc")

   Sample Output
   -------------
   List(List(46L, 50L, 70L), List(37L, 42L, 51L), List(33L, 58L), List(61L, 87L), List(8L, 21L, 79L))
 */

import io.shiftleft.codepropertygraph.Cpg
import io.shiftleft.codepropertygraph.generated.nodes.{Call, Expression, FieldIdentifier, Identifier, Literal, Method}
import io.shiftleft.codepropertygraph.generated.{Operators, nodes}
import io.shiftleft.semanticcpg.language._
import io.shiftleft.semanticcpg.language.operatorextension._
import overflowdb.traversal._

import java.io.{File}
import java.io.{PrintWriter, File => JFile}
import scala.io.Source

/* POINT 1: FC -- API/Library Function Call
   Characteristics: 
     + is an instance of the node property: Call
     + not Operations call (operations, i.e. +, is identified as Call type in joern)
     + name in the set of Sensitive Functions, sensitive_func.pkl
 */
private def getFCPoints(funcsPath: String): List[Long] = {
	val lines = Source.fromFile(funcsPath).getLines().toIterator
	var funcs: Array[String] = Array()
	// Actually sensi_funcs.txt only contains 1 line
	while (lines.hasNext) {
		val line = lines.next()
		funcs = line.substring(2, line.length-2).split("', '")
	}

	cpg.call
		.filter(call => !call.name.startsWith("<operator>"))
		.filter(call => funcs.exists({x: String => x == call.name}))
		.id.l
}

/* POINT 2: AU -- Array Use; PU -- Pointer Use
   They have same Characteristics, so we put them together as POINT 2
   Characteristics:
     + is an instance of the node property: ArrayAccess
 */
private def getAUPUPoints(): List[Long] = {
	cpg.arrayAccess.id.l
}

/* POINT 3: AE -- Arithmetic Expression
   Characteristics:
     + a expression contains operations, i.e. addition, subtraction, assignmentPlus etc.
     + is an instance of the node property: arithmetic
 */
private def getAEPoints(): List[Long] = {
	cpg.arithmetic.id.l
}

/* POINT 4: FP -- Function Parameter
   Characteristics:
     + is an instance of the node property: parameter
     + has property lineNumber
 */
private def getFPPoints(): List[Long] = {
	cpg.parameter.filter(_.lineNumber != None).id.l
}

/* POINT 5: FR -- Function Return Statement
   Characteristics:
     + is an instance of the node property: ret
 */
private def getFRPoints(): List[Long] = {
	cpg.ret.id.l
}

private def getAllPointsID(funcsPath: String): List[List[Long]] = {
	var list: List[List[Long]] = Nil
	((((list:+getFCPoints(funcsPath)):+getAUPUPoints()):+getAEPoints()):+getFPPoints()):+getFRPoints()
}

def getPoints(dir_path: String, funcs_path: String) = {
	// The same with which in the extract-funcs-info.sc
	val dirPath = dir_path
	val funcsPath = funcs_path
	val points = getAllPointsID(funcsPath)
	val filename = "AllVulPoints.txt"
	
	var item = 0
	val writer = new PrintWriter(new JFile(dirPath + "/" + filename))
	for (item <- points) {
		writer.println(item)
	}
	writer.close()
}
