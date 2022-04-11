package com.juliohenrique.reconhecimento_facial_mlkit


interface SimilarityClassifier {

    class Recognition(var id: String, var title: String, var distance: Float) {

        private var extra: Any? = null

        fun setExtra(extra: Any?) {
            this.extra = extra
        }

        fun getExtra(): Any? {
            return extra
        }
        override fun toString(): String {
            var resultString = ""
            if (id != null) {
                resultString += "[$id] "
            }
            if (title != null) {
                resultString += "$title "
            }
            if (distance != null) {
                resultString += String.format("(%.1f%%) ", distance * 100.0f)
            }
            return resultString.trim { it <= ' ' }
        }
    }
}