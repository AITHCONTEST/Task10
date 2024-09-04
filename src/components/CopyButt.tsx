import copyImg from "./../assets/copy.svg";
import styles from "./styles/button.module.scss";
import React from "react";

interface CopyButtProps {
    targetRef: React.RefObject<HTMLTextAreaElement>;
}

const CopyButt = ({ targetRef }: CopyButtProps) => {
    const tap = () => {
        if (targetRef.current) {
            targetRef.current.select();
            navigator.clipboard.writeText(targetRef.current.value)
                .then(() => {
                    console.log("Text copied to clipboard");
                })
                .catch((err) => {
                    console.error("Failed to copy text: ", err);
                });
        }
    };

    return (
        <div onClick={tap} className={styles.butt}>
            <img src={copyImg} alt="<->" />
        </div>
    );
};

export default CopyButt;
