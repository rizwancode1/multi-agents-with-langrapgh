import coreWebVitals from "eslint-config-next/core-web-vitals";
import nextTypescript from "eslint-config-next/typescript";

const eslintConfig = [
  {
    ignores: [
      "node_modules/**",
      ".next/**",
      "out/**",
      "next-env.d.ts",
      "*.tsbuildinfo",
    ],
  },
  ...coreWebVitals,
  ...nextTypescript,
  {
    files: ["contexts/ThemeContext.tsx"],
    rules: {
      // Theme hydration intentionally reads localStorage and syncs the DOM
      // class in an effect: lazy state init would cause hydration text
      // mismatches for theme-dependent markup. suppressHydrationWarning on
      // <html>/<body> covers the flash.
      "react-hooks/set-state-in-effect": "off",
    },
  },
];

export default eslintConfig;
