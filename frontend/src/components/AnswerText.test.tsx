import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AnswerText } from "./AnswerText";

afterEach(cleanup);

describe("AnswerText", () => {
  it("turns each valid citation, including grouped ones, into a button", () => {
    render(<AnswerText text="Lowers LDL [1]. Dose matters [2, 3]." sourceCount={3} onCite={() => {}} />);
    expect(screen.getAllByRole("button").map((b) => b.textContent)).toEqual(["1", "2", "3"]);
  });

  it("calls onCite with the clicked source number", async () => {
    const onCite = vi.fn();
    render(<AnswerText text="Claim [2]." sourceCount={3} onCite={onCite} />);
    await userEvent.click(screen.getByRole("button", { name: "Go to source 2" }));
    expect(onCite).toHaveBeenCalledWith(2);
  });

  it("shows a citation to a missing source as a non-clickable marker", () => {
    render(<AnswerText text="Claim [9]." sourceCount={3} onCite={() => {}} />);
    expect(screen.queryByRole("button")).toBeNull();
    expect(screen.getByTitle(/not retrieved/i).textContent).toBe("9");
  });

  it("keeps the surrounding text intact", () => {
    const { container } = render(
      <AnswerText text="Before [1] after." sourceCount={1} onCite={() => {}} />,
    );
    expect(container.textContent).toBe("Before 1 after.");
  });
});

describe("bold markers", () => {
  it("renders **text** as bold without showing asterisks", () => {
    const { container } = render(
      <AnswerText text="**Answer:** fasting helps [1]." sourceCount={1} onCite={() => {}} />,
    );
    expect(container.querySelector("strong")?.textContent).toBe("Answer:");
    expect(container.textContent).not.toContain("**");
  });
});

describe("fullwidth citation brackets", () => {
  it("turns 【1】 into a clickable citation", () => {
    render(<AnswerText text={"Statins help【1】."} sourceCount={1} onCite={() => {}} />);
    expect(screen.getByRole("button", { name: "Go to source 1" })).toBeTruthy();
  });
});
