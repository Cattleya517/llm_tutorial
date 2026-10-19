// Predict-the-behavior quiz. Markup:
// <div class="quiz" data-answer="1">          (0-based index of the correct button)
//   <p class="q">question</p>
//   <button>A</button><button>B</button><button>C</button>
//   <p class="why">explanation shown after any click</p>
// </div>
document.querySelectorAll(".quiz").forEach((quiz) => {
  const answer = Number(quiz.dataset.answer);
  const buttons = [...quiz.querySelectorAll("button")];
  buttons.forEach((btn, i) => {
    btn.addEventListener("click", () => {
      if (quiz.classList.contains("done")) return;
      quiz.classList.add("done");
      btn.classList.add(i === answer ? "ok" : "bad");
      buttons[answer].classList.add("ok");
      buttons.forEach((b) => (b.disabled = true));
    });
  });
});
