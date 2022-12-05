/* extract-funcs-info.scala

   This script extracts cpg info and ast node info in .dot format for each function.
	 The results would be written into .txt and .json format files, for further operations.

   Input: A valid CPG

   Running the Script
   ------------------
   Use joern command line tool
	 Run command: cpg.runScript("extract-funcs-info.sc")

 */

import io.shiftleft.codepropertygraph.Cpg
import io.shiftleft.codepropertygraph.generated.nodes.{Call, Expression, FieldIdentifier, Identifier, Literal, Method}
import io.shiftleft.codepropertygraph.generated.{Operators, nodes}
import io.shiftleft.semanticcpg.language._
import io.shiftleft.semanticcpg.language.operatorextension._
import overflowdb.traversal._

import java.io.{File}
import java.io.{PrintWriter, File => JFile}

private def listFuncs(): List[Method] = {
	// cpg.method.filter(m => m.name != "<global>" && !m.name.startsWith("<operator>")).l
  cpg.method.filter(m => !m.isExternal).l
}

private def extractNodes(outputDir: String="results") = {
	val outPath = outputDir + "/NodesForAll.json"
	cpg.all.toJson |> outPath
}

private def extractFuncCpg(outputDir: String="results") = {
	val outDir = outputDir + "/cpgs/"
	val outPath = new File(outDir)
	outPath.mkdirs()

	val list = listFuncs()
	var item = 0
	for (item <- list if item.filename != "<empty>") {
		// Extract cpgs in .dot format if a function has source file
		val filenames = item.filename.split("/")
		val filename = filenames(filenames.length-1)
		val name = filename.substring(0, filename.lastIndexOf(".c"))
		val savename = name + ".txt"
		val savePath = outDir + savename

		item.dotCpg14.l |> savePath
	}
}

def extractFuncs(dir_path: String) = {
	// Optional: modify the output path
	val dirPath = dir_path
	val resultPath = new File(dirPath)
	resultPath.mkdirs()

	extractNodes(dirPath)
	println("Extract all cpg nodes over...")
	extractFuncCpg(dirPath)
	println("Extract all function cpgs over...")
}
